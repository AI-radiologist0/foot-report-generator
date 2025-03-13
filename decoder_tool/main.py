import argparse
import os
import pickle
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import wandb
import logging
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu
from torch.utils.tensorboard import SummaryWriter

import _init_path
from config import cfg, update_config
from dataset.joint_patches import FootPatchesDataset
from utils.collate_fn import TokenizedReportCollateFn
from utils.utils import EarlyStopping, BestModelSaver, get_optimizer
from core.decoder_trainer import DecoderTrainer

import models
from models.decoder import get_decoder_with_image_embedding, DecoderWithLoRA

def initialize_logger(output_dir):
    """Initialize a logger to save logs to a file and print them to the console."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler(os.path.join(output_dir, 'training.log'))
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

def calculate_bleu_score(references, predictions):
    """
    Calculate BLEU score for the generated reports.

    Args:
        references: List of reference texts (ground truth).
        predictions: List of generated texts.

    Returns:
        Average BLEU score across all samples.
    """
    scores = []
    for ref, pred in zip(references, predictions):
        score = sentence_bleu([ref.split()], pred.split())
        scores.append(score)
    return sum(scores) / len(scores)

def generate_texts(model, tokenizer, dataloader, device, feature_extractor):
    """
    Generate texts using the model for a given dataloader, including image inputs.

    Args:
        model: The trained decoder model.
        tokenizer: The tokenizer used for encoding/decoding.
        dataloader: Dataloader containing the input data (including images).
        device: Device to run the model on (CPU or GPU).
        feature_extractor: The feature extractor for generating image embeddings.

    Returns:
        List of generated texts and reference texts.
    """
    model.eval()
    feature_extractor.eval()  # Ensure feature extractor is in evaluation mode
    generated_texts = []
    reference_texts = []

    with torch.no_grad():
        for batch in dataloader:
            # Extract text inputs
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Extract and process images
            images = batch["images"].to(device)  # Assuming "images" key in dataset
            patches = batch["patch_tensors"].to(device)
            image_embeddings = feature_extractor(images, patches)  # Generate image embeddings

            # Generate text using the model with image embeddings
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=50,
                image_embeddings=image_embeddings  # Pass image embeddings to the model
            )

            # Decode generated texts
            generated_texts.extend([tokenizer.decode(output, skip_special_tokens=True) for output in outputs])

            # Decode reference texts
            reference_texts.extend([tokenizer.decode(label, skip_special_tokens=True) for label in batch["labels"]])

    return generated_texts, reference_texts

def main():
    # Argument parser for configuration file
    parser = argparse.ArgumentParser(description="Training pipeline for Foot Report Generator")
    parser.add_argument('--cfg', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()

    # Update configuration
    update_config(cfg, args)

    # Set up output directory and logger
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join('output', f"decoder_{timestamp}")
    logger = initialize_logger(output_dir)

    # Log configuration
    logger.info("Configuration:")
    logger.info(cfg)

    # Device setup
    device = torch.device('cuda' if cfg.DEVICE == 'GPU' and torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    writer_dict = {'writer': SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard')),
                   "train_global_steps": 0,
                   "valid_global_steps": 0}

    # Initialize Wandb
    run = wandb.init(
        config=dict(cfg),
        name=f"Decoder_{cfg.DECODER.NAME}_{cfg.DATASET.PKL}_{cfg.TRAIN.END_EPOCH}_epochs",
        notes="Training decoder model with LoRA",
        tags=["decoder", "LoRA", "training"]
    )

    if cfg.DECODER.NAME == 'GPT2':
        community = "openai-community"
    
    decoder_huggingface_name = f"{community}/{cfg.DECODER.NAME.lower()}"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(decoder_huggingface_name)
    logger.info("GPT-2 Tokenizer loaded.")

    # Add a padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        logger.info("Added [PAD] token as the pad_token to the tokenizer.")
    
    tokenizer.padding_side = "left"

    # Load dataset
    with open(cfg.DATASET.PKL, 'rb') as f:
        pkl_data = pickle.load(f)

    dataset = FootPatchesDataset(cfg, pkl_data)
    collate_fn = TokenizedReportCollateFn(tokenizer, max_length=cfg.DECODER.EXTRA.MAX_SEQ_LENGTH)

    # Split dataset into train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU, shuffle=False, collate_fn=collate_fn)

    logger.info("Dataset split into train, validation, and test sets.")

    # Load feature extractor model
    feature_extractor_name = cfg.MODEL.NAME
    feature_extractor = eval('models.' + feature_extractor_name + '.get_feature_extractor')(
        cfg, is_train=True, remove_classifier=True
    )
    feature_extractor = feature_extractor.to(device)
    logger.info("Feature extractor model loaded.")

    # # Generate image embeddings
    # logger.info("Generating image embeddings...")
    # image_embeddings = generate_image_embeddings(feature_extractor, train_loader, device)

    # LoRA configuration
    lora_config = LoraConfig(
        r=8,  # LoRA rank
        lora_alpha=32,  # Scaling factor
        target_modules=["c_attn", "image_projection"],  # LoRA applied modules
        lora_dropout=0.1,  # Dropout
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Initialize decoder model with image embedding support
    decoder_model = get_decoder_with_image_embedding(decoder_huggingface_name)

    # Resize token embeddings to accommodate new tokens
    decoder_model.resize_token_embeddings(len(tokenizer))
    logger.info("Resized model embeddings to accommodate new tokens.")

    # Apply LoRA to the decoder model
    logger.info("Applying LoRA to the decoder model...")
    lora_wrapped_model = get_peft_model(decoder_model, lora_config)

    # Wrap the LoRA model with the custom wrapper to support image_embeddings
    decoder_model = DecoderWithLoRA(lora_wrapped_model, lora_config)
    logger.info("LoRA successfully applied.")

    # Move the model to the target device
    decoder_model.to(device)
    logger.info("Decoder model with LoRA initialized.")

    # Decoder Trainer
    optimizer = get_optimizer(cfg, decoder_model)
    trainer = DecoderTrainer(cfg, decoder_model, tokenizer, feature_extractor, writer_dict=writer_dict)
    early_stopping = EarlyStopping(verbose=True)
    best_model_saver = BestModelSaver(verbose=True)
    
    # Training loop
    for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        logger.info(f"Epoch {epoch} starting...")
               
        train_loss = trainer.train(epoch, train_loader, optimizer)
        val_loss, val_avg_bleu_score = trainer.validate(epoch, val_loader)

        logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}")
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_bleu": val_avg_bleu_score})

        best_model_saver.save(decoder_model, val_loss)
        early_stopping(val_loss)

        if early_stopping:
            logger.info("Early stopping triggered.")
            break

    # Generate texts and calculate BLEU score for test set
    test_generated, test_references = generate_texts(decoder_model, tokenizer, test_loader, device)
    test_bleu_score = calculate_bleu_score(test_references, test_generated)
    logger.info(f"Test BLEU Score = {test_bleu_score:.4f}")

    # Finish Wandb run
    run.finish()
    logger.info("Training complete.")

if __name__ == '__main__':
    main()
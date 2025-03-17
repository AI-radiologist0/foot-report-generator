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
from core.decoder_trainer import DecoderTrainer, clean_special_tokens

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
                max_length=150,
                image_embeddings=image_embeddings,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,  # Ensure EOS token is used
                early_stopping=True  # Stop generation when EOS token is encountered
            )

            # Decode generated texts
            raw_generated_texts = [tokenizer.decode(output, skip_special_tokens=False) for output in outputs]
            processed_generated_texts = [clean_special_tokens(text) for text in raw_generated_texts]
            generated_texts.extend(processed_generated_texts)

            # Decode reference texts
            raw_reference_texts = [tokenizer.decode(label, skip_special_tokens=False) for label in batch["labels"]]
            processed_reference_texts = [clean_special_tokens(text) for text in raw_reference_texts]
            reference_texts.extend(processed_reference_texts)

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

    # Tokenizer setup
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        logger.info("Added [PAD] token as the pad_token to the tokenizer.")
    

    if tokenizer.eos_token != "[EOS]":
        tokenizer.add_special_tokens({'eos_token': "[EOS]"})
        logger.info("Added [EOS] token as the eos_token to the tokenizer.")
    
    if tokenizer.bos_token != "[BOS]":
        tokenizer.add_special_tokens({'bos_token': '[BOS]'})
        logger.info("Added [BOS] token as the bos_token to the tokenizer.")
    
    special_tokens_dict = {
            "additional_special_tokens": ["[FINDING]", "[CONCLUSION]", "[RECOMMEND]"]
        }
    tokenizer.add_special_tokens(special_tokens_dict)

    tokenizer.padding_side = "left"

    # Load dataset
    with open(cfg.DATASET.PKL, 'rb') as f:
        pkl_data = pickle.load(f)

    dataset = FootPatchesDataset(cfg, pkl_data)
    collate_fn = TokenizedReportCollateFn(tokenizer, max_length=cfg.DECODER.EXTRA.MAX_SEQ_LENGTH, save_path='./token.txt')

    # Logging Special Token IDs.
    logger.info(f"Special Tokens: {tokenizer.special_tokens_map}")
    logger.info(f"BOS Token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    logger.info(f"EOS Token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    logger.info(f"PAD Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    logger.info(f"Additional Special Tokens: {tokenizer.additional_special_tokens}")
    logger.info(f"Additional Special Token IDs: {tokenizer.convert_tokens_to_ids(tokenizer.additional_special_tokens)}")

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

    # Log PAD Token ID
    logger.info(f"Decoder Model PAD Token ID: {decoder_model.config.pad_token_id}")
    decoder_model.config.pad_token_id = tokenizer.pad_token_id
    logger.info(f"Decoder Model Updated PAD Token ID: {decoder_model.config.pad_token_id}")

    decoder_model.config.eos_token_id = tokenizer.eos_token_id
    logger.info(f"Set Decoder Model EOS Token ID to: {decoder_model.config.eos_token_id}")
    
    # Log BOS Token ID
    decoder_model.config.bos_token_id = tokenizer.bos_token_id
    logger.info(f"Set Decoder Model BOS Token ID to: {decoder_model.config.bos_token_id}")

    logger.info(f"Tokenizer PAD Token ID: {tokenizer.pad_token_id}, Model PAD Token ID: {decoder_model.config.pad_token_id}")
    logger.info(f"Tokenizer EOS Token ID: {tokenizer.eos_token_id}, Model EOS Token ID: {decoder_model.config.eos_token_id}")
    logger.info(f"Tokenizer BOS Token ID: {tokenizer.bos_token_id}, Model BOS Token ID: {decoder_model.config.bos_token_id}")
   
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
    
    # When cfg.PHASE == 'train', train the model
    if cfg.PHASE == 'train':
        # Training loop
        for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
            logger.info(f"Epoch {epoch} starting...")
                
            train_loss = trainer.train(epoch, train_loader, optimizer)
            val_loss, val_avg_bleu_score, sample_predictions = trainer.validate(epoch, val_loader)

            if sample_predictions:
                columns = ["sample_id", "reference", "prediction", "bleu_score"]
                data = []
                for i, sample in enumerate(sample_predictions):
                    data.append([i, sample['reference'], sample['prediction'], sample['bleu_score']])
                
                wandb.log({"sample_predictions": wandb.Table(columns=columns, data=data)})

            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}")
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_bleu": val_avg_bleu_score})

            best_model_saver.save(decoder_model, val_loss)
            early_stopping(val_loss)

            if early_stopping:
                logger.info("Early stopping triggered.")
                break

    # Test the model on the test set when cfg.PHASE == 'test' or cfg.PHASE == 'train'
    if cfg.PHASE == 'test' or cfg.PHASE == 'train':

        saved_path = cfg.DECODER.EXTRA.PT if cfg.PHASE == 'test' else best_model_saver.save_path
        
        # Load the best model
        best_model = best_model_saver.load_best_model(decoder_model, saved_path)
        best_model.eval()
        best_model.to(device)
        logger.info("Best model loaded.")
        

        # Generate texts and calculate BLEU score for test set
        test_generated, test_references = generate_texts(decoder_model, tokenizer, test_loader, device, feature_extractor)
        test_bleu_score = calculate_bleu_score(test_references, test_generated)
        logger.info(f"Test BLEU Score = {test_bleu_score:.4f}")

    # Finish Wandb run
    run.finish()
    logger.info("Training complete.")

if __name__ == '__main__':
    main()
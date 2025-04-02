import argparse
import gc
import os
import pickle
import torch
import wandb
import logging
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split


import _init_path
from config import cfg, update_config
from dataset.joint_patches import FootPatchesDataset
from utils.collate_fn import TokenizedReportCollateFn
from utils.tokenizer import load_and_prepare_tokenizer
from utils.utils import EarlyStopping, BestModelSaver, get_optimizer
from core.decoder_trainer import DecoderTrainer, clean_special_tokens

import models
from models.decoder import get_decoder_with_image_embedding, DecoderWithLoRA
from models.decoder import get_decoder


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
            image_embeddings = feature_extractor(images, patches)   # Generate image embeddings

            
            
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
            # raw_reference_texts = [tokenizer.decode(label, skip_special_tokens=False) for label in batch["labels"]]
            raw_reference_texts = [
                    tokenizer.decode([token for token in label.tolist() if token >= 0], skip_special_tokens=False)
                    for label in batch["labels"]
                ]
            processed_reference_texts = [clean_special_tokens(text) for text in raw_reference_texts]
            reference_texts.extend(processed_reference_texts)

    return generated_texts, reference_texts

def main():
    # Argument parser for configuration file
    parser = argparse.ArgumentParser(description="Training pipeline for Foot Report Generator")
    parser.add_argument('--cfg', type=str, default='config/decoder/train_decoder_without_LoRA.yaml', required=False, help='Path to the configuration file')
    args = parser.parse_args()

    # Update configuration
    update_config(cfg, args)

    # Empty Cache
    torch.cuda.empty_cache()

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
    use_lora = "w.LoRA" if cfg.DECODER.USE_LORA else "w/o.LoRA"
    # Initialize Wandb
    run = wandb.init(
        config=dict(cfg),
        name=f"Decoder_{cfg.DECODER.NAME}_{cfg.DATASET.PKL}_{cfg.TRAIN.END_EPOCH}_epochs",
        notes="Training decoder model with LoRA",
        tags=["decoder", use_lora , "training"]
    )

    if cfg.DECODER.NAME == 'GPT2':
        community = "openai-community"
    if cfg.DECODER.NAME == "meerkat-7b-v1.0":
        community = "dmis-lab"
    if cfg.DECODER.NAME == "dmis-lab-meerkat-7b-v1.0-AWQ-4bit-smashed":
        community = "PrunaAI"
    if cfg.DECODER.NAME == "EXAONE-3.5-2.4B-Instruct":
        community = "LGAI-EXAONE"
    
    model_size = '' if cfg.DECODER.EXTRA.MODEL_SIZE == "small" else f"-{cfg.DECODER.EXTRA.MODEL_SIZE.lower()}"

    decoder_huggingface_name = f"{community}/{cfg.DECODER.NAME.lower()}{model_size}"
    logging.info(f"Model Name: {decoder_huggingface_name}")
    tokenizer = load_and_prepare_tokenizer(
        model_name = decoder_huggingface_name,
        additional_special_tokens= ['<img>', "[FINDING]", "[CONCLUSION]", "[RECOMMEND]", "[DIAGNOSIS]", "[RECOMMENDATION]"]
    )
    
    # Load dataset
    with open(cfg.DATASET.PKL, 'rb') as f:
        pkl_data = pickle.load(f)

    dataset = FootPatchesDataset(cfg, pkl_data)
    collate_fn = TokenizedReportCollateFn(tokenizer, cfg, max_length=cfg.DECODER.EXTRA.MAX_SEQ_LENGTH, save_path='./meerkat_token.txt')

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

    decoder_model = get_decoder(cfg, tokenizer, model_name=decoder_huggingface_name)

    logger.info(f"Decoder Model Class: {type(decoder_model)} on {decoder_huggingface_name}")
    # total_params = sum(p.numel() for p in decoder_model.parameters() if p.requires_grad)
    # logger.info(f"Total trainable parameters: {total_params}")

    decoder_model = decoder_model.to(device)
    logger.info(f"Configuration: Decoder Model and to device on {device}")

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
            val_loss = trainer.validate(epoch, val_loader)

            if epoch % cfg.PRINT_FREQ == 0 and epoch > 0:
                bleu, rouge, bert, predictions = trainer.inference(epoch, val_loader)
                if predictions:
                    columns = ["sample_id", "reference", "prediction", "bleu_score", "ROUGE-L", "BERTSCORE"]
                    data = []
                    for i, sample in enumerate(predictions):
                        data.append([
                            i, 
                            sample['reference'], 
                            sample['prediction'], 
                            sample['bleu_score'],
                            sample['rouge_l'], 
                            sample['bert_f1']]
                        )
                    
                    wandb.log({"sample_predictions": wandb.Table(columns=columns, data=data)})
                wandb.log({"BLUE": bleu, "ROUGE": rouge, "bert": bert})
                logger.info(f"Inference Step {epoch}: BLEU_SCORE = {bleu:.4f} ROUGE = {rouge:.4f} BERT = {bert:.4f}")


            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}")
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})


            best_model_saver.save(decoder_model, val_loss)
            early_stopping(val_loss)

            if early_stopping:
                logger.info("Early stopping triggered.")
                break
    
    gc.collect()
    torch.cuda.empty_cache()

    # Test the model on the test set when cfg.PHASE == 'test' or cfg.PHASE == 'train'
    if cfg.PHASE == 'test' or cfg.PHASE == 'train':
        
        saved_path = cfg.DECODER.EXTRA.PT if cfg.PHASE == 'test' and cfg.DECODER.EXTRA.PT is not None else best_model_saver.save_path
        
        # Load the best model
        best_model = best_model_saver.load_best_model(decoder_model, saved_path)
        best_model.eval()
        best_model.to(device)
        logger.info("Best model loaded.")
        
        bleu, rouge, bert, predictions = trainer.inference(epoch, val_loader)
        if predictions:
            columns = ["sample_id", "reference", "prediction", "bleu_score", "ROUGE-L", "BERTSCORE"]
            data = []
            for i, sample in enumerate(predictions):
                data.append([
                        i, 
                        sample['reference'], 
                        sample['prediction'], 
                        sample['bleu_score'],
                        sample['rouge_l'], 
                        sample['bert_f1']]
                    )
            
            wandb.log({"sample_predictions(TEST)": wandb.Table(columns=columns, data=data)})
        wandb.log({"bleu(TEST)": bleu, "rouge(TEST)": rouge, "bert": bert})
        logger.info(f"BLEU Score(TEST) = {bleu:.4f}")
        
        # # Generate texts and calculate BLEU score for test set
        # test_generated, test_references = generate_texts(decoder_model, tokenizer, test_loader, device, feature_extractor)
        # test_bleu_score = calculate_bleu_score(test_references, test_generated)
        # logger.info(f"Test BLEU Score = {test_bleu_score:.4f}")

    gc.collect()
    torch.cuda.empty_cache()

    # Finish Wandb run
    run.finish()
    logger.info("Training complete.")

if __name__ == '__main__':
    main()
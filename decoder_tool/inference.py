# decoder_tool/inference.py
import argparse
import os
import torch
import pickle
import gc
import logging
import wandb
from datetime import datetime
from torch.utils.data import DataLoader, random_split

import _init_path
from config import cfg, update_config
from dataset.joint_patches import FootPatchesDataset
from utils.collate_fn import TokenizedReportCollateFn
from utils.tokenizer import load_and_prepare_tokenizer
from core.decoder_trainer import DecoderTrainer, clean_special_tokens
from utils.utils import BestModelSaver
import models
from models.decoder import get_decoder

def initialize_logger(output_dir):
    """Initialize logger for inference."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler(os.path.join(output_dir, 'inference.log'))
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger



def inference_model(args):
    # Update configuration
    update_config(cfg, args)
    torch.cuda.empty_cache()

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join('output', f"inference_{timestamp}")
    logger = initialize_logger(output_dir)
    logger.info("Configuration:")
    logger.info(cfg)

    device = torch.device('cuda' if cfg.DEVICE == 'GPU' and torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Initialize Wandb (optional)
    run = wandb.init(
        config=dict(cfg),
        name=f"Inference_{cfg.DECODER.NAME}_{cfg.DATASET.PKL}",
        notes="Running inference for decoder model"
    )

    # Determine community for model name
    if cfg.DECODER.NAME == 'GPT2':
        community = "openai-community"
    elif cfg.DECODER.NAME == "meerkat-7b-v1.0":
        community = "dmis-lab"
    elif cfg.DECODER.NAME == "dmis-lab-meerkat-7b-v1.0-AWQ-4bit-smashed":
        community = "PrunaAI"
    elif cfg.DECODER.NAME == "EXAONE-3.5-2.4B-Instruct":
        community = "LGAI-EXAONE"
    else:
        community = "default"

    model_size = '' if cfg.DECODER.EXTRA.MODEL_SIZE == "small" else f"-{cfg.DECODER.EXTRA.MODEL_SIZE.lower()}"
    decoder_huggingface_name = f"{community}/{cfg.DECODER.NAME.lower()}{model_size}"
    logger.info(f"Model Name: {decoder_huggingface_name}")

    tokenizer = load_and_prepare_tokenizer(
        model_name=decoder_huggingface_name,
        additional_special_tokens=['<img>', "[FINDING]", "[CONCLUSION]", "[RECOMMEND]", "[DIAGNOSIS]", "[RECOMMENDATION]"]
    )

    # Load dataset and create test_loader
    with open(cfg.DATASET.PKL, 'rb') as f:
        pkl_data = pickle.load(f)
    dataset = FootPatchesDataset(cfg, pkl_data)
    collate_fn = TokenizedReportCollateFn(tokenizer, cfg, max_length=cfg.DECODER.EXTRA.MAX_SEQ_LENGTH, save_path='./meerkat_token.txt')
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    _, _, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU, shuffle=False, collate_fn=collate_fn)
    logger.info("Test dataset prepared.")

    # Load feature extractor model
    feature_extractor_name = cfg.MODEL.NAME
    feature_extractor = eval('models.' + feature_extractor_name + '.get_feature_extractor')(
        cfg, is_train=False, remove_classifier=True
    )
    feature_extractor = feature_extractor.to(device)
    logger.info("Feature extractor model loaded.")    

    # Setup BestModelSaver and load best model via trainer's safe update method
    best_model_saver = BestModelSaver(verbose=True)
    saved_path = cfg.DECODER.EXTRA.PT
    # Load best model: 내부에서 trainer.update_model을 호출하므로 기존 모델은 삭제되고 GPU 캐시가 정리됨
    best_model = get_decoder(cfg, tokenizer, model_name=decoder_huggingface_name)
    best_model = best_model_saver.load_best_model(best_model, saved_path)
    trainer = DecoderTrainer(cfg, best_model, tokenizer, feature_extractor,
                             writer_dict={"writer": None, "train_global_steps": 0, "valid_global_steps": 0})
    best_model.eval()
    logger.info("Best model loaded for inference.")

    # Run inference on test set
    bleu, rouge, bert, predictions = trainer.inference(0, test_loader)
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
                sample['bert_f1']
            ])
        wandb.log({"sample_predictions(TEST)": wandb.Table(columns=columns, data=data)})
    wandb.log({"bleu(TEST)": bleu, "rouge(TEST)": rouge, "bert(TEST)": bert})
    logger.info(f"BLEU Score(TEST) = {bleu:.4f}")

    run.finish()
    logger.info("Inference complete.")
    # Clean up
    del trainer.model
    torch.cuda.empty_cache()
    gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Inference pipeline for Foot Report Generator")
    parser.add_argument('--cfg', type=str, default='config/decoder/train_decoder_without_LoRA.yaml', required=False,
                        help='Path to the configuration file')
    parser.add_argument('--pt', type=str, default=None, help='Path to the best model checkpoint')
    args = parser.parse_args()

    inference_model(args)


if __name__ == '__main__':
    main()

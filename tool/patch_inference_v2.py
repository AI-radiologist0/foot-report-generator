import argparse
import logging
import os
import pickle
from datetime import datetime
import pprint

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import wandb

import _init_path
from config import cfg, update_config
from dataset.joint_patches import FootPatchesDataset
from core.patch_trainer import PatchTrainer
from utils.utils import BestModelSaver
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Inference classification result.')
    parser.add_argument('--cfg', help='path to the config file', required=True, type=str)
    return parser.parse_args()


def create_logger(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(os.path.join(output_dir, 'inference.log'))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def prepare_test_loader(cfg):
    with open(cfg.DATASET.PKL, 'rb') as f:
        data = pickle.load(f)
    dataset = FootPatchesDataset(cfg, data)
    test_size = int(0.15 * len(dataset))
    test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - test_size, test_size])[1]
    test_loader = DataLoader(test_dataset, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU, shuffle=False)
    return test_loader, dataset


def prepare_subset(dataset, indices):
    return Subset(dataset, indices)

def set_classifier(model, dim):
    model.classifier = nn.Sequential(
        nn.Linear(dim, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )
    return model


def main():
    args = parse_args()
    update_config(cfg, args)
    device = torch.device('cuda' if cfg.DEVICE == 'GPU' else 'cpu')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join('output', f'inference_{timestamp}')
    logger = create_logger(out_dir)
    logger.info(pprint.pformat(cfg))

    writer_dict = {'writer': SummaryWriter(log_dir=os.path.join(out_dir, 'tensorboard')),
                "train_global_steps": 0,
                "valid_global_steps": 0}

    run = wandb.init(project='foot-arthritis-infer', config=dict(cfg))

    model = eval('models.' + cfg.MODEL.NAME + '.get_feature_extractor')(
        cfg, is_train=False
    )
    model = set_classifier(model, 4110)

    test_loader, full_dataset = prepare_test_loader(cfg)
    best_model_saver = BestModelSaver(verbose=True)
    model = best_model_saver.load_best_model(model, cfg.MODEL.EXTRA.CKPT).to(device)
    trainer = PatchTrainer(cfg, model=model, output_dir=out_dir, writer_dict=writer_dict)

    metrics, preds, labels = trainer.inference(1, model, test_loader, device)
    logger.info("Main stage results:")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")

    normal_indices = [i for i, label in enumerate(labels) if label == 1]
    logger.info(f"Proceeding to second stage with {len(normal_indices)} normal samples...")

    del model, trainer
    torch.cuda.empty_cache()

    second_stage(normal_indices, full_dataset, device, logger, writer_dict)

    run.finish()


def second_stage(normal_indices, full_dataset, device, logger, writer_dict):
    # Import second config
    second_cfg_path = 'config/large/inference/large_oa_normal_padding_use_pkl_infer.yaml'
    update_config(cfg, argparse.Namespace(cfg=second_cfg_path))

    

    model = eval('models.' + cfg.MODEL.NAME + '.get_feature_extractor')(
        cfg, is_train=False
    )
    model = set_classifier(model, 4110)

    subset = prepare_subset(full_dataset, normal_indices)
    loader = DataLoader(subset, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU, shuffle=False)

    model = eval('models.' + cfg.MODEL.NAME + '.get_feature_extractor')(
        cfg, is_train=False
    )
    model = set_classifier(model, 4110)

    best_model_saver = BestModelSaver(verbose=True)
    model = best_model_saver.load_best_model(model, cfg.MODEL.EXTRA.CKPT).to(device)
    trainer = PatchTrainer(cfg, model=model, output_dir='output/second_stage', writer_dict=writer_dict)

    metrics, preds, labels = trainer.inference(1, model, loader, device)
    logger.info("Second stage results:")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")


if __name__ == '__main__':
    main()

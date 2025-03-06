# ------------------------------------------------------------------------------
# # Licensed under the MIT License.
# Written by Jeongmin Kim (jm.kim@dankook.ac.kr)
# ------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from collections import OrderedDict
from datetime import datetime
import logging
import os
import pickle
import pprint

import wandb
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
import torch.utils.data as torchUtils
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data.distributed
import torchvision.transforms as transforms


import _init_path
# from core.config import config
# from core.config import update_config
# from core.config import update_dir
from config import cfg, update_config
from core.patch_trainer import PatchTrainer
from dataset.joint_patches import FootPatchesDataset

import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train classification Network with feature extractor')
    # general
    parser.add_argument('--cfg',
                        default='config/test.yaml',
                        help='experiment configure file name',
                        required=False,
                        type=str)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.use_detect_bbox:
        config.TEST.USE_GT_BBOX = not args.use_detect_bbox
    if args.flip_test:
        config.TEST.FLIP_TEST = args.flip_test
    if args.post_process:
        config.TEST.POST_PROCESS = args.post_process
    if args.shift_heatmap:
        config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file
    if args.coco_bbox_file:
        config.TEST.COCO_BBOX_FILE = args.coco_bbox_file

def create_logger(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(os.path.join(output_dir, 'training.log'))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    return logger

def main():
    args = parse_args()
    update_config(cfg, args)

    
    # set up timestamp, config name, target_classes and branch model
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    target_classes = cfg.DATASET.TARGET_CLASSES
    str_target_classes = '_'.join(target_classes)
    # two branch model 
    # local = config.MODEL.LOCAL
    # global = config.MODEL.GLOBAL

    # Set output directories
    final_output_dir = os.path.join('output', f"twobranchModel_{timestamp}_{str_target_classes}_classifier")
    ckpt_save_dir = os.path.join(final_output_dir, 'ckpt')
    os.makedirs(ckpt_save_dir, exist_ok=True)

    device = torch.device('cuda' if cfg.DEVICE == 'GPU' else 'cpu')
    
    logger = create_logger(final_output_dir)

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(cfg))

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.' + cfg.MODEL.NAME + '.get_feature_extractor')(
        cfg, is_train=True
    )

    model = model.to(device)
    
    writer_dict = {'writer': SummaryWriter(log_dir=os.path.join(final_output_dir, 'tensorboard')),
                   "train_global_steps": 0,
                   "valid_global_steps": 0}
    
    # Wandb Init
    run = wandb.init(
        project="classification(2branchModel)",
        config = dict(cfg),
        name=f"twobranchModel_{timestamp}_{str_target_classes}_classifier",  # Include timestamp and cfg name in run name
        notes="This run includes best_model and final_model logging to WandB.",  # Optional description
        tags=["hand-arthritis", "classification(test)"]  # Optional tags
    )


    # Data loading code
    with open(cfg.DATASET.PKL, 'rb') as f:
        pkl_data = pickle.load(f)

    logging.info('Dataset Loading ...')
    dataset = FootPatchesDataset(cfg, pkl_data)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torchUtils.random_split(dataset, [train_size, val_size, test_size])
    train_loader = torchUtils.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torchUtils.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = torchUtils.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Train and Validation by Using Trainier
    logging.info('Prepare Trainer ...')
    trainer = PatchTrainer(cfg, model=model, output_dir=final_output_dir, writer_dict=writer_dict)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    logging.info("Start Training")
    for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        train_loss, train_acc = trainer.train(epoch, train_loader, optimizer=optimizer)
        val_perf, val_loss, precision, recall, f1 = trainer.validate(epoch, model, val_loader)
        
        # logging data (Wandb)
        wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_performance": val_perf,
                       "precision": precision, "recall": recall, "f1_score": f1, "epoch": epoch}, step=epoch)
        # early stopping
        
        
    # Test
    test_perf, test_loss, precision, recall, f1 = trainer.validate(1, model, test_loader)
    run.log({"test_perf": test_perf, "test_loss": test_loss, 
                       "precision(test)": precision, "recall(test)": recall, "f1_score(test)": f1})

    run.finish()

if __name__ == '__main__':
    main()

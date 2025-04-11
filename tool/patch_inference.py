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
from torch import nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
import torch.utils.data as torchUtils
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data.distributed
import torchvision.transforms as transforms


import _init_path
from config import cfg, update_config
from core.patch_trainer import PatchTrainer
from dataset.joint_patches import FootPatchesDataset, FootPatchesDatasetWithJson
from utils.utils import EarlyStopping, BestModelSaver

import models

torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description='Inference classification result.')
    # general
    parser.add_argument('--cfg',
                        default='config/large/inference/large_abnormal_normal_padding_use_pkl_infer.yaml',
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

from torch.utils.data import Subset

def prepare_subset_by_class(dataset, class_list):
    """
    특정 클래스만 포함된 서브셋 생성
    Args:
        dataset: 전체 dataset
        class_list: 선택할 클래스 이름 리스트 (예: ['ra', 'normal'])
    Returns:
        torch.utils.data.Subset
    """
    indices = []
    for idx, entry in enumerate(dataset.data):
        label = entry['class'].lower()
        if label in class_list:
            indices.append(idx)
    return Subset(dataset, indices)

def load_model(cfg, device):
    model = eval('models.' + cfg.MODEL.NAME + '.get_feature_extractor')(cfg, is_train=False)
    checkpoint = torch.load(cfg.MODEL.EXTRA.CKPT, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])  # 'state_dict' 키로 저장돼 있다고 가정
    model.to(device)
    return model

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


    # set up timestamp, config name, target_classes and branch model
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    target_classes = cfg.DATASET.TARGET_CLASSES
    str_target_classes = '_'.join(target_classes)
    # two branch model 
    raw = cfg.MODEL.EXTRA.RAW
    patch = cfg.MODEL.EXTRA.PATCH
    model_name = cfg.MODEL.NAME
    dataset_scale = 'large' if cfg.DATASET.PKL == 'data/pkl/output200x300.pkl' else 'small'
    dataset_scale = 'large' if cfg.DATASET.PKL == 'data/pkl/final_output_left_right_ordered.pkl' else 'small'

    # Set output directories
    final_output_dir = os.path.join('output', f"twobranchModel_{timestamp}_{str_target_classes}_classifier")
    ckpt_save_dir = os.path.join(final_output_dir, 'ckpt')
    os.makedirs(ckpt_save_dir, exist_ok=True)

    device = torch.device('cuda' if cfg.DEVICE == 'GPU' else 'cpu')
    
    logger = create_logger(final_output_dir)

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(cfg))

    logger.info(
        f"Timestamp: {timestamp}, "
        f"Target Classes: {str_target_classes}, "
        f"Model Name: {model_name}, "
        f"Dataset Scale: {dataset_scale}, "
        f"Raw Branch: {raw}, "
        f"Patch Branch: {patch}"
    )

    # if the phase is not inference, main function shut down
    if cfg.PHASE not in ['infer', 'inference']:
        logger.info("This script needs for inference phase.")
        return


    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.' + model_name + '.get_feature_extractor')(
        cfg, is_train=False
    )
    model = set_classifier(model, 4110)
    
    
    writer_dict = {'writer': SummaryWriter(log_dir=os.path.join(final_output_dir, 'tensorboard')),
                   "train_global_steps": 0,
                   "valid_global_steps": 0}
    
    branch_type = "4branchModel" if model_name == 'feature_extractor2' else "2branchModel"

    # Wandb Init
    run = wandb.init(
        project=f"classification({branch_type}_infer)",
        config = dict(cfg),
        name=f"{branch_type}_{model_name}_{dataset_scale}_{timestamp}_{str_target_classes}_classifier_raw_{raw}_patch_{patch}",  # Include timestamp and cfg name in run name
        notes="This run includes best_model and final_model logging to WandB.",  # Optional description
        tags=["foot-arthritis", "classification(test)"]  # Optional tags
    )

    if cfg.DATASET.USE_PKL:
        # Data loading code
        with open(cfg.DATASET.PKL, 'rb') as f:
            pkl_data = pickle.load(f)
        dataset = FootPatchesDataset(cfg, pkl_data)
    else:
        dataset = FootPatchesDatasetWithJson(cfg)
    logging.info('Dataset Loading ...')

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    _, _, test_dataset = torchUtils.random_split(dataset, [train_size, val_size, test_size])
    test_loader = torchUtils.DataLoader(test_dataset, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU, shuffle=False)
    best_model_saver = BestModelSaver(verbose=True)
    model = best_model_saver.load_best_model(model, cfg.MODEL.EXTRA.CKPT).to(device)
    
    trainer = PatchTrainer(cfg, model=model, output_dir=final_output_dir, writer_dict=writer_dict)
    

    # 4. 추론 및 metric 계산
    metrics, preds, labels = trainer.inference(1, model, test_loader, device)

    # 5. 결과 출력
    # logging.info(f"📊 Results for [{task}] vs normal")
    for k, v in metrics.items():
        logging.info(f"{k:<10}: {v:.4f}")
    
       
    
    # logging.info("Start Inference")
    
    run.finish()

if __name__ == '__main__':
    main()

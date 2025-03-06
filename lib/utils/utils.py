# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import random
import time
from pathlib import Path


import torch
import torch.optim as optim
from tqdm import tqdm

from core.config import get_model_name


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.PKL 
    dataset = dataset.replace('/', '_')
    model, _ = get_model_name(cfg)
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / model / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
        (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def count_labels(data, target_class):
    """
        Method: count images for each labels.
        
        Param:
        data: data from pkl
        target_class(list): target class
        
        return:
        class_count(dict): the number of images for each labels
    
    """
    from collections import defaultdict
    class_counts = defaultdict(int)
    data_by_class = defaultdict(list)

    # if isinstance(data, list):
    #     return Counter(entry['class'].lower for entry in data_list)

    
    for key, entry in tqdm(data.items(), desc="counting dataset"):
        class_label = entry.get('class', '').lower()
        if class_label in target_class and os.path.exists(entry['file_path']):
            class_counts[class_label] += 1
            data_by_class[class_label].append(entry)

    return class_counts, data_by_class

# Prepare binary data
def prepare_binary_data(data, target_class, normal_class='normal'):
    """
        Method: to prepare binary data with augmentation.
        
        Param:
        data: data from pkl
        target_class(dict): target class label to use. ex) {'oa': 0, 'normal': 0}
        
        return:
        balanced_data (Equal the number of Each class label)
        count_class(dict): the number of original each class
        
    """
    class_counts, data_by_class = count_labels(data, target_class)
    logging.info(f"class_count: {class_counts}")
    balanced_data = []
    min_class_count = min(class_counts.values())
    for label in class_counts.keys():
        sampled_data = random.sample(data_by_class[label], min_class_count)
        balanced_data.extend(sampled_data)
    
    augmented_data = balanced_data * 2
    
    return augmented_data
    
    

def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )

    return optimizer


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth.tar'))


class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                
                
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + val * n
        self.count = self.count + n
        self.avg = self.sum / self.count if self.count != 0 else 0
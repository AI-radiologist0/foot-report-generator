# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
import os
import logging
import random
import time
from pathlib import Path
import tempfile
import tarfile


from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torch
import torch.optim as optim
import wandb
from tqdm import tqdm

from core.config import get_model_name

random.seed(42)


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


def count_labels(data, target_class, cfg):
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
        class_label = entry.get('class', "").lower() if cfg.DATASET.USE_PKL else entry.get('class_label', "").lower()
        if class_label == "":
            logging.warning(f"Missing class label for entry: {key}")
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
    
    
def prepare_abnormal_normal_data(data, cfg):
    """
    Prepare dataset for abnormal vs normal classification.

    Args:
        data (dict): Data from pickle file.
        cfg (object): Configuration object containing:
            - cfg.DATASET.BALANCE (bool): If True, balances dataset by equalizing class sample counts.
            - cfg.DATASET.AUGMENT (bool): If True, applies augmentation by duplicating balanced data.

    Returns:
        dataset (list): Processed dataset (balanced/augmented based on cfg).
        class_counts (dict): Number of images per class before processing.
        final_counts (dict): Number of images per class after processing.
    """
    random.seed(42)

    # Define abnormal and normal classes
    abnormal_classes = ['ra', 'oa', 'gout']
    normal_classes = ['normal']

    target_classes = abnormal_classes + normal_classes

    # Count labels and group data by class
    class_counts, data_by_class = count_labels(data, target_classes, cfg)
    logging.info(f"Original class distribution: {class_counts}")

    # Combine abnormal classes into a single "abnormal" class
    combined_data_by_class = {
        'abnormal': sum([data_by_class[cls] for cls in abnormal_classes], []),
        'normal': data_by_class['normal']
    }

    combined_class_counts = {
        'abnormal': sum([class_counts[cls] for cls in abnormal_classes]),
        'normal': class_counts['normal']
    }

    logging.info(f"Combined class distribution: {combined_class_counts}")

    # If neither balancing nor augmentation is enabled, return raw data
    if not cfg.DATASET.BALANCE and not cfg.DATASET.AUGMENT:
        return sum(combined_data_by_class.values(), []), combined_class_counts, combined_class_counts

    # Determine minimum class count for balancing
    min_class_count = min(combined_class_counts.values())

    # Balance dataset by sampling the same number of instances from each class
    balanced_data = []
    final_counts = {}

    for label in combined_data_by_class.keys():
        sampled_data = random.sample(combined_data_by_class[label], min_class_count)
        balanced_data.extend(sampled_data)
        final_counts[label] = min_class_count

    logging.info(f"Balanced class distribution: {final_counts}")

    # Apply data augmentation if enabled
    if cfg.DATASET.AUGMENT:
        balanced_data = balanced_data * 2  # Duplicate dataset
        logging.info("Applied data augmentation: dataset size doubled.")

    return balanced_data, combined_class_counts, final_counts

def prepare_data(data, target_classes, cfg, is_binary=False):
    """
    Prepare dataset for either binary or multi-class classification with optional balancing and augmentation.

    Args:
        data (dict): Data from pickle file.
        target_classes (list): List of target class labels. Ex) ['oa', 'normal'] or ['ra', 'oa', 'gout', 'normal']
                               Special case: ['abnormal', 'normal'] will use abnormal vs normal processing.
        cfg (object): Configuration object containing:
            - cfg.DATASET.BINARY (bool): If True, prepares for binary classification.
            - cfg.DATASET.BALANCE (bool): If True, balances dataset by equalizing class sample counts.
            - cfg.DATASET.AUGMENT (bool): If True, applies augmentation by duplicating balanced data.
        is_binary (bool): Deprecated, use cfg.DATASET.BINARY instead.

    Returns:
        dataset (list): Processed dataset (balanced/augmented based on cfg).
        class_counts (dict): Number of images per class before processing.
        final_counts (dict): Number of images per class after processing.
    """
    random.seed(42)
    
    # Check if we're doing abnormal vs normal classification
    if len(target_classes) == 2 and 'abnormal' in target_classes and 'normal' in target_classes:
        logging.info("Using abnormal vs normal processing logic")
        return prepare_abnormal_normal_data(data, cfg)
    
    # Binary classification check
    if is_binary and len(target_classes) != 2:
        raise ValueError("Binary classification requires exactly 2 target classes.")

    # Regular processing for other class combinations
    class_counts, data_by_class = count_labels(data, target_classes, cfg)
    logging.info(f"Original class distribution: {class_counts}")

    # If neither balancing nor augmentation is enabled, return raw data
    if not cfg.DATASET.BALANCE and not cfg.DATASET.AUGMENT:
        return sum(data_by_class.values(), []), class_counts, class_counts  

    # Determine minimum class count for balancing
    min_class_count = min(class_counts.values())

    # Balance dataset by sampling the same number of instances from each class
    balanced_data = []
    final_counts = {}

    for label in target_classes:
        sampled_data = random.sample(data_by_class[label], min_class_count)
        balanced_data.extend(sampled_data)
        final_counts[label] = min_class_count

    logging.info(f"Balanced class distribution: {final_counts}")

    # Apply data augmentation if enabled
    if cfg.DATASET.AUGMENT:
        balanced_data = balanced_data * 2  # Duplicate dataset
        logging.info("Applied data augmentation: dataset size doubled.")

    return balanced_data, class_counts, final_counts


def prepare_abnormal_normal_data_with_seed(data, cfg, seed: int = 42):
    random.seed(seed)
    abnormal_classes = ['ra', 'oa', 'gout']
    normal_classes = ['normal']

    target_classes = abnormal_classes + normal_classes
    class_counts, data_by_class = count_labels(data, target_classes, cfg)
    logging.info(f"[Seed={seed}] Original class distribution: {class_counts}")

    combined_data_by_class = {
        'abnormal': sum([data_by_class[cls] for cls in abnormal_classes], []),
        'normal': data_by_class['normal']
    }

    combined_class_counts = {
        'abnormal': sum([class_counts[cls] for cls in abnormal_classes]),
        'normal': class_counts['normal']
    }

    if not cfg.DATASET.BALANCE and not cfg.DATASET.AUGMENT:
        return sum(combined_data_by_class.values(), []), combined_class_counts, combined_class_counts

    min_class_count = min(combined_class_counts.values())

    balanced_data = []
    final_counts = {}
    for label in combined_data_by_class:
        sampled_data = random.sample(combined_data_by_class[label], min_class_count)
        balanced_data.extend(sampled_data)
        final_counts[label] = min_class_count

    if cfg.DATASET.AUGMENT:
        balanced_data = balanced_data * 2
        logging.info(f"[Seed={seed}] Data augmentation applied")

    return balanced_data, combined_class_counts, final_counts


def prepare_data_with_seed(data, target_classes, cfg, seed: int = 42):
    """
    prepare_data()와 동일한 로직을 사용하되, 랜덤 시드를 제어할 수 있도록 확장한 버전.
    abnormal vs normal 분류도 그대로 지원.
    """
    random.seed(seed)

    # abnormal vs normal 분류
    if len(target_classes) == 2 and 'abnormal' in target_classes and 'normal' in target_classes:
        logging.info(f"[Seed={seed}] Using abnormal vs normal processing logic")
        return prepare_abnormal_normal_data_with_seed(data, cfg, seed)

    # 일반 클래스 처리
    class_counts, data_by_class = count_labels(data, target_classes, cfg)
    logging.info(f"[Seed={seed}] Original class distribution: {class_counts}")

    if not cfg.DATASET.BALANCE and not cfg.DATASET.AUGMENT:
        return sum(data_by_class.values(), []), class_counts, class_counts

    min_class_count = min(class_counts.values())

    balanced_data = []
    final_counts = {}
    for label in target_classes:
        sampled_data = random.sample(data_by_class[label], min_class_count)
        balanced_data.extend(sampled_data)
        final_counts[label] = min_class_count

    if cfg.DATASET.AUGMENT:
        balanced_data = balanced_data * 2
        logging.info(f"[Seed={seed}] Data augmentation applied")

    return balanced_data, class_counts, final_counts


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

    elif cfg.TRAIN.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WD
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
                logging.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
    def __bool__(self):
        return self.early_stop
        
                
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
        

class BestModelSaver:
    def __init__(self, save_path="best_model.pth.tar.gz", verbose=False):
        """
        Args:
            save_path (str): Path to save the best model (as tar.gz)
            verbose (bool): Whether to print log messages
        """
        self.save_path = os.path.join(wandb.run.dir, save_path)
        self.final_path = os.path.join(wandb.run.dir, "final_model.pth.tar.gz")
        self.best_loss = float("inf")
        self.verbose = verbose
        self.prev_save_path = None

    def save(self, model, val_loss):
        """
        Args:
            model (torch.nn.Module): Model to save
            val_loss (float): Current validation loss
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            # Delete previous best tar.gz file if exists
            if self.prev_save_path and os.path.exists(self.prev_save_path):
                os.remove(self.prev_save_path)
            # Save state_dict to a temporary .pth file
            with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_pth:
                torch.save(model.state_dict(), tmp_pth.name)
                tmp_pth_path = tmp_pth.name
            # Compress the .pth file into tar.gz
            with tarfile.open(self.save_path, "w:gz") as tar:
                tar.add(tmp_pth_path, arcname="best_model.pth")
            os.remove(tmp_pth_path)
            self.prev_save_path = self.save_path
            if self.verbose:
                logging.info(f"🔹 Best model saved as tar.gz! New best loss: {val_loss:.6f}")

    def save_final_model(self, model):
        # Save final model as tar.gz
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_pth:
            torch.save(model.state_dict(), tmp_pth.name)
            tmp_pth_path = tmp_pth.name
        with tarfile.open(self.final_path, "w:gz") as tar:
            tar.add(tmp_pth_path, arcname="final_model.pth")
        os.remove(tmp_pth_path)
        if self.verbose:
            logging.info(f"✅ Final model saved to {self.final_path}")

    def load_best_model(self, model, path=None):
        """Load the best model from a tar.gz file."""
        import tarfile
        import tempfile
        tar_path = path if path is not None else self.save_path
        with tarfile.open(tar_path, "r:gz") as tar:
            member = tar.getmember("best_model.pth") if "best_model.pth" in tar.getnames() else tar.getmembers()[0]
            with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_pth:
                tmp_pth.write(tar.extractfile(member).read())
                tmp_pth_path = tmp_pth.name
        model.load_state_dict(torch.load(tmp_pth_path))
        os.remove(tmp_pth_path)
        if self.verbose:
            print(f"✅ Best model loaded from {tar_path}")
        return model

def insert_before_bos(input_ids, input_embeddings, image_embeddings, bos_token_id, attention_mask=None, labels=None):
    """
    삽입 위치: BOS 토큰 앞
    input_embeddings: (B, L, D)
    image_embeddings: (B, D)
    Returns: combined_embeddings, new_attention_mask, new_labels
    """
    B, L = input_ids.shape
    D = input_embeddings.shape[-1]
    device = input_ids.device

    # dtype alignment
    image_embeddings = image_embeddings.to(dtype=input_embeddings.dtype)

    # BOS 위치: 각 배치에서 첫 번째 bos_token_id의 위치
    bos_pos = (input_ids == bos_token_id).float().argmax(dim=1)  # (B,)

    # 새 텐서 초기화
    new_len = L + 1
    new_input_embeddings = torch.zeros((B, new_len, D), dtype=input_embeddings.dtype, device=device)
    new_attention_mask = torch.zeros((B, new_len), dtype=torch.long, device=device) if attention_mask is not None else None
    new_labels = torch.full((B, new_len), -100, dtype=torch.long, device=device) if labels is not None else None

    # 시프트 인덱스 계산
    arange_L = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    shifted_indices = torch.where(arange_L < bos_pos.unsqueeze(1), arange_L, arange_L + 1)  # (B, L)

    # input_embeddings 채우기
    new_input_embeddings.scatter_(1, shifted_indices.unsqueeze(-1).expand(-1, -1, D), input_embeddings)
    new_input_embeddings[torch.arange(B), bos_pos] = image_embeddings

    # attention_mask
    if attention_mask is not None:
        new_attention_mask.scatter_(1, shifted_indices, attention_mask)
        new_attention_mask[torch.arange(B), bos_pos] = 1

    # labels
    if labels is not None:
        new_labels.scatter_(1, shifted_indices, labels)
        new_labels[torch.arange(B), bos_pos] = -100

    return new_input_embeddings, new_attention_mask, new_labels


def prepare_generate_inputs(img_token_id, bos_token_id, device, batch_size=1):
    """
    Returns input_ids and attention_mask shaped (B, 2) with [<img>, <bos>]
    """
    input_ids = torch.tensor(
        [[bos_token_id]] * batch_size,
        dtype=torch.long,
        device=device
    )
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask

def extract_labels(dataset):
    labels = []
    for i in range(len(dataset)):
        label = dataset[i][2]

        if isinstance(label, torch.Tensor):
            label = int(label.item())  # 텐서에서 스칼라 정수로 변환

        labels.append(label)
    return labels


def stratified_split_dataset(dataset, seed=42):
    labels = dataset.get_labels()

    train_idx, temp_idx = train_test_split(
        range(len(dataset)), test_size=0.2, stratify=labels, random_state=seed
    )
    temp_labels = [labels[i] for i in temp_idx]
    train_idx, temp_idx = train_test_split(
        range(len(dataset)), test_size=0.2, stratify=labels, random_state=seed
    )
    temp_labels = [labels[i] for i in temp_idx]

    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=seed
    )

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


from collections import Counter

def check_label_distribution_from_subset(subset, name=""):
    base_dataset = subset.dataset
    indices = subset.indices

    if not hasattr(base_dataset, 'get_labels'):
        raise AttributeError("Subset의 원본 dataset에 get_labels() 메서드가 없습니다.")

    all_labels = base_dataset.get_labels()
    subset_labels = [all_labels[i] for i in indices]

    dist = Counter(subset_labels)
    logging.info(f"\n📊 {name} 분포:")
    for cls, count in sorted(dist.items()):
        logging.info(f"  클래스 {cls}: {count}개")

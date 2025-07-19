import os
import json
import logging
import torch
import torch.utils.data as torchUtils
import numpy as np
from datetime import datetime
from torch.optim.adamw import AdamW
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Subset
from torchvision import transforms

import _init_path
from config import cfg, update_config
from dataset.joint_patches import FinalSamplesDataset
from utils.utils import (
    EarlyStopping, BestModelSaver, check_label_distribution_from_subset, 
    split_dataset_by_patient_id_and_class, get_conservative_xray_augmentation, 
    get_improved_xray_augmentation, get_xray_augmentation
)
from utils.vis import plot_roc_curve
from core.patch_trainer import PatchTrainer
import models
import wandb

def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler
    file_handler = logging.FileHandler(os.path.join(output_dir, 'augmentation_experiment.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def run_augmentation_experiment(seed: int, cfg, model_name, final_output_dir, device, 
                               augmentation_type, augment_ratio, use_patch=False):
    """
    Augmentation 실험 실행
    
    Args:
        augmentation_type: 'none', 'conservative', 'improved', 'strong'
        augment_ratio: 0, 0.5, 1, 2
        use_patch: True면 RAW + Patch 증강, False면 RAW만 증강
    """
    cfg.defrost()
    cfg.DATASET.SEED = seed
    cfg.freeze()

    # 1. transform 정의
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # 2. 전체 데이터셋 생성
    dataset = FinalSamplesDataset(cfg, image_transform=val_test_transform)

    # 3. 환자ID 기준 split
    data_list = list(dataset.data.values())
    train_data, val_data, test_data = split_dataset_by_patient_id_and_class(
        data_list, val_size=0.2, test_size=0.2, random_state=seed
    )

    # 4. 인덱스 추출
    def get_indices_from_entries(dataset, entries):
        entry_set = set((e['patient_id'], e['file_path']) for e in entries)
        indices = [
            idx for idx, v in dataset.data.items()
            if (v['patient_id'], v['file_path']) in entry_set
        ]
        return indices

    train_indices = get_indices_from_entries(dataset, train_data)
    val_indices = get_indices_from_entries(dataset, val_data)
    test_indices = get_indices_from_entries(dataset, test_data)

    # 5. Augmentation 설정
    if augmentation_type == 'none':
        augment_transform = None
        augment_ratio = 0
    elif augmentation_type == 'conservative':
        augment_transform = get_conservative_xray_augmentation()
    elif augmentation_type == 'improved':
        augment_transform = get_improved_xray_augmentation()
    elif augmentation_type == 'strong':
        augment_transform = get_xray_augmentation()
    else:
        raise ValueError(f"Unknown augmentation_type: {augmentation_type}")

    # 6. 데이터셋 생성
    train_dataset = FinalSamplesDataset(
        cfg, 
        image_transform=train_transform,
        augment_transform=augment_transform,
        augment_ratio=augment_ratio
    )
    
    val_dataset = FinalSamplesDataset(
        cfg, 
        image_transform=val_test_transform,
        augment_transform=None,
        augment_ratio=0
    )
    
    test_dataset = FinalSamplesDataset(
        cfg, 
        image_transform=val_test_transform,
        augment_transform=None,
        augment_ratio=0
    )

    # 7. Subset 생성
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    test_subset = Subset(test_dataset, test_indices)

    # 8. 로깅
    logging.info(f"\n🔬 [Augmentation Experiment]")
    logging.info(f"Type: {augmentation_type}")
    logging.info(f"Ratio: {augment_ratio}")
    logging.info(f"Use Patch: {use_patch}")
    
    check_label_distribution_from_subset(train_subset, "Train")
    check_label_distribution_from_subset(val_subset, "Validation")
    check_label_distribution_from_subset(test_subset, "Test")

    logging.info(f"Train size: {len(train_subset)}")
    logging.info(f"Val size: {len(val_subset)}")
    logging.info(f"Test size: {len(test_subset)}")

    # 9. DataLoader 생성
    train_loader = torchUtils.DataLoader(train_subset, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU, shuffle=True, num_workers=4)
    val_loader = torchUtils.DataLoader(val_subset, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU, num_workers=4)
    test_loader = torchUtils.DataLoader(test_subset, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU, num_workers=4)

    # 10. 모델 생성
    model = eval('models.' + model_name + '.get_feature_extractor')(cfg, is_train=True).to(device)
    logging.info(f"✅ Loaded model: {model.__class__.__name__}")

    # 11. WandB 설정
    run_name = f"{model_name}_aug_{augmentation_type}_ratio_{augment_ratio}_patch_{use_patch}_seed{seed}"
    
    run = wandb.init(
        project="augmentation-experiment",
        name=run_name,
        config={
            "augmentation_type": augmentation_type,
            "augment_ratio": augment_ratio,
            "use_patch": use_patch,
            "seed": seed,
            **dict(cfg)
        },
        reinit=True
    )

    # 12. 학습
    writer_dict = {
        'writer': SummaryWriter(log_dir=os.path.join(final_output_dir, f'tensorboard/{run_name}')),
        'train_global_steps': 0,
        'valid_global_steps': 0
    }

    trainer = PatchTrainer(cfg, model=model, output_dir=final_output_dir, writer_dict=writer_dict)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    early_stopping = EarlyStopping()
    best_model_saver = BestModelSaver()

    for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        train_loss, train_acc = trainer.train(epoch, train_loader, optimizer)
        val_perf, val_loss, precision, recall, f1, _, _ = trainer.validate(epoch, model, val_loader)
        best_model_saver.save(model, val_loss)
        early_stopping(val_loss)
        
        run.log({
            "train_loss": train_loss, 
            "train_acc": train_acc, 
            "val_loss": val_loss,
            "val_perf": val_perf, 
            "precision": precision, 
            "recall": recall, 
            "f1": f1, 
            "epoch": epoch
        })
        
        if early_stopping:
            logging.info("Early stopping triggered.")
            break

    # 13. 테스트
    model = best_model_saver.load_best_model(model).to(device).eval()
    test_perf, test_loss, precision, recall, f1, y_true, y_score = trainer.validate(999, model, test_loader)
    
    run.log({
        "test_perf": test_perf, 
        "test_loss": test_loss,
        "precision(test)": precision, 
        "recall(test)": recall, 
        "f1_score(test)": f1
    })
    
    # 14. 결과 저장
    result = {
        "augmentation_type": augmentation_type,
        "augment_ratio": augment_ratio,
        "use_patch": use_patch,
        "seed": seed,
        "test_perf": test_perf,
        "test_loss": test_loss,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    run.finish()
    
    return result

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='config/large/tmp/proj/swin_t_resnet/origin_ra_normal_sampling20_proj_linear.yaml', type=str)
    parser.add_argument('--phase', default='1', type=str, choices=['1', '2'], 
                       help='Phase 1: RAW only, Phase 2: RAW + Patch')
    args = parser.parse_args()

    update_config(cfg, args)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_name = cfg.MODEL.NAME
    target_classes = cfg.DATASET.TARGET_CLASSES
    str_target_classes = '_'.join(target_classes)
    
    final_output_dir = os.path.join('output', f"augmentation_exp_{timestamp}_{model_name}")
    logger = setup_logger(final_output_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 실험 설정
    if args.phase == '1':
        # Phase 1: RAW Image Augmentation 실험
        experiments = [
            # 1. Augmentation ratio 변화 실험
            {'type': 'conservative', 'ratio': 0, 'patch': False},
            {'type': 'conservative', 'ratio': 0.5, 'patch': False},
            {'type': 'conservative', 'ratio': 1, 'patch': False},
            {'type': 'conservative', 'ratio': 2, 'patch': False},
            
            # 2. 보수적 증강 실험
            {'type': 'none', 'ratio': 0, 'patch': False},
            {'type': 'conservative', 'ratio': 1, 'patch': False},
            
            # 3. 향상된 증강 실험
            {'type': 'improved', 'ratio': 1, 'patch': False},
            {'type': 'strong', 'ratio': 1, 'patch': False},
        ]
    else:
        # Phase 2: RAW + Patch Augmentation 실험
        experiments = [
            {'type': 'conservative', 'ratio': 1, 'patch': True},
            {'type': 'improved', 'ratio': 1, 'patch': True},
            {'type': 'strong', 'ratio': 1, 'patch': True},
        ]
    
    # 결과 저장
    all_results = []
    
    for i, exp in enumerate(experiments):
        logging.info(f"\n{'='*60}")
        logging.info(f"Experiment {i+1}/{len(experiments)}")
        logging.info(f"Type: {exp['type']}, Ratio: {exp['ratio']}, Patch: {exp['patch']}")
        logging.info(f"{'='*60}")
        
        # 1번 실험 (seed=42)
        result = run_augmentation_experiment(
            seed=42, cfg=cfg, model_name=model_name, 
            final_output_dir=final_output_dir, device=device,
            augmentation_type=exp['type'], 
            augment_ratio=exp['ratio'], 
            use_patch=exp['patch']
        )
        
        logging.info(f"Seed 42: Acc={result['test_perf']:.4f}, F1={result['f1']:.4f}")
        
        # 결과 저장
        all_results.append(result)
    
    # 최종 결과 저장
    results_file = os.path.join(final_output_dir, 'augmentation_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logging.info(f"\n✅ All experiments completed!")
    logging.info(f"Results saved to: {results_file}")
    
    # 결과 요약
    logging.info(f"\n📊 Final Summary:")
    for result in all_results:
        logging.info(f"{result['augmentation_type']}_ratio{result['augment_ratio']}_patch{result['use_patch']}: "
                    f"Acc={result['test_perf']:.4f}, F1={result['f1']:.4f}")

if __name__ == '__main__':
    main() 
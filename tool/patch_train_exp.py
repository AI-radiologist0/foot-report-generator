import os
import json
import logging
import torch
import torch.utils.data as torchUtils
import numpy as np
from datetime import datetime
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset
from torchvision import transforms

import _init_path
from config import cfg, update_config
from dataset.joint_patches import FinalSamplesDataset
from utils.utils import EarlyStopping, BestModelSaver, check_label_distribution_from_subset, split_dataset_by_patient_id_and_class
from utils.vis import plot_roc_curve
from core.patch_trainer import PatchTrainer
import models
import wandb
from wandb import AlertLevel

def flatten_binary_scores(y_score_list):
    return [float(s[0]) if isinstance(s, (list, np.ndarray)) and len(s) == 1 else float(s)
            for s in y_score_list]


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler
    file_handler = logging.FileHandler(os.path.join(output_dir, 'training.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def run_experiment(seed: int, cfg, model_name, final_output_dir, device, tags, run_name):
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

    # 2. 전체 데이터셋 생성 (인덱스 추출용)
    dataset = FinalSamplesDataset(cfg, image_transform=val_test_transform)

    # 3. 리스트로 변환 및 환자ID 기준 split
    data_list = list(dataset.data.values())
    train_data, val_data, test_data = split_dataset_by_patient_id_and_class(
        data_list, val_size=0.2, test_size=0.2, random_state=seed
    )

    # 4. 인덱스 추출 함수
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

    # 5. subset별로 transform 다르게 적용
    # train: 증강 포함 (1:1 비율)
    train_dataset = FinalSamplesDataset(
        cfg, 
        image_transform=train_transform,
        augment_transform=None,
        augment_ratio=1  # 1:1 비율로 증강
    )
    
    # val/test: 증강 없음
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

    # 6. Subset 생성
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    test_subset = Subset(test_dataset, test_indices)

    check_label_distribution_from_subset(train_subset, "Train")
    check_label_distribution_from_subset(val_subset, "Validation")
    check_label_distribution_from_subset(test_subset, "Test")

    logging.info(f"Train size: {len(train_subset)} (원본 + 증강본)")
    logging.info(f"Val size: {len(val_subset)}")
    logging.info(f"Test size: {len(test_subset)}")

    train_loader = torchUtils.DataLoader(train_subset, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU, shuffle=True, num_workers=4)
    val_loader = torchUtils.DataLoader(val_subset, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU, num_workers=4)
    test_loader = torchUtils.DataLoader(test_subset, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU, num_workers=4)

    # concat_patch 디버그용 첫 샘플 이미지 저장
    if getattr(cfg.DATASET, 'CONCAT_PATCH', False):
        import torchvision.transforms as T
        import os
        os.makedirs('log', exist_ok=True)
        for batch in train_loader:
            patches = batch[1]  # (B, 1, 224, 224)
            patch_img = patches[0]  # (1, 224, 224)
            patch_img = patch_img.clone()
            patch_img = (patch_img - patch_img.min()) / (patch_img.max() - patch_img.min() + 1e-8)
            patch_img = (patch_img * 255).byte()
            pil_img = T.ToPILImage()(patch_img)
            pil_img.save("log/concat_patch_debug.png")
            print("✅ concat_patch 첫 샘플 이미지를 log/concat_patch_debug.png로 저장했습니다.")
            break

    model = eval('models.' + model_name + '.get_feature_extractor')(cfg, is_train=True).to(device)
    logging.info(f"✅ Loaded model: {model.__class__.__name__}")
    model.log_model_info(cfg, logging)

    writer_dict = {
        'writer': SummaryWriter(log_dir=os.path.join(final_output_dir, f'tensorboard/seed{seed}')),
        'train_global_steps': 0,
        'valid_global_steps': 0
    }

    run = wandb.init(
        project="classification(final-sample-data)",
        name=run_name,
        config=dict(cfg),
        tags=tags,
        reinit=True
    )

    trainer = PatchTrainer(cfg, model=model, output_dir=final_output_dir, writer_dict=writer_dict)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    early_stopping = EarlyStopping()
    best_model_saver = BestModelSaver()

    

    for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        train_loss, train_acc = trainer.train(epoch, train_loader, optimizer)
        val_perf, val_loss, precision, recall, f1, _, _ = trainer.validate(epoch, model, val_loader)
        best_model_saver.save(model, val_loss)
        early_stopping(val_loss)
        run.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss,
                 "val_perf": val_perf, "precision": precision, "recall": recall, "f1": f1, "epoch": epoch})
        if early_stopping:
            logging.info("Early stopping triggered.")
            break

    model = best_model_saver.load_best_model(model).to(device).eval()
    test_perf, test_loss, precision, recall, f1, y_true, y_score = trainer.validate(999, model, test_loader)
    
    run.log({"test_perf": test_perf, "test_loss": test_loss,
             "precision(test)": precision, "recall(test)": recall, "f1_score(test)": f1})
    
    y_score = flatten_binary_scores(y_score) if trainer.is_binary else y_score
    
    roc_path = plot_roc_curve(
        y_true=y_true,
        y_score=y_score,
        output_dir=final_output_dir,
        seed=seed,
        is_binary=trainer.is_binary,
        run=run
    )
    
    wandb.alert(
        title=f"[{run_name}] Experiment Finished", 
        text=f"Seed {seed} Finish. \n Accuracy: {test_perf:.4f}, F1: {f1:.4f}", 
        level=AlertLevel.INFO
    )

    run.finish()

    return test_perf, f1


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='config/large/tmp/origin_oa_normal.yaml', type=str)
    parser.add_argument('--repeat', default=20, type=int)
    args = parser.parse_args()

    update_config(cfg, args)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_name = cfg.MODEL.NAME
    target_classes = cfg.DATASET.TARGET_CLASSES
    str_target_classes = '_'.join(target_classes)
    final_output_dir = os.path.join('output', f"multi_run_{timestamp}_{model_name}")
    logger = setup_logger(final_output_dir)
    repeat = args.repeat

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    extra = cfg.MODEL.EXTRA
    freeze = cfg.MODEL.FREEZE
    tags = [
        "foot-arthritis",
        f"MODEL={model_name}",
        f"TARGET={str_target_classes}",
        f"RAW={extra.RAW}",
        f"PATCH={extra.PATCH}",
        f"WITH_ATTN={extra.WITH_ATTN}",
        f"ONLYCAT={extra.ONLYCAT}",
        f"VIEWCAT={extra.VIEWCAT}",
        f"FREEZE_BACKBONE={freeze.BACKBONE}",
        f"FREEZE_CLASSIFIER={freeze.CLASSIFIER}",
        f"FREEZE_PROJECTION={freeze.PROJECTION}",
        f"REPEAT_NUMBER={repeat}",
        f"CONCAT_PATCH={cfg.DATASET.CONCAT_PATCH}"
    ]
    
    logging.info("📌 Experiment Tags:")
    for tag in tags:
        logging.info(f"  - {tag}")
        
    logging.info(f"MODEL NAME: {model_name}")

    concat_patch_str = "concatpatch" if cfg.DATASET.CONCAT_PATCH else "patchseq"

    if getattr(cfg.DATASET, 'MULTI_RUN', False):
        accs, f1s = [], []
        for seed in range(repeat):
            run_name = f"{model_name}_{str_target_classes}_{concat_patch_str}_seed{seed}_{timestamp}"
            logging.info(f"Running experiment {seed}...")
            acc, f1 = run_experiment(seed, cfg, model_name, final_output_dir, device, tags, run_name)
            accs.append(acc)
            f1s.append(f1)
            logging.info(f"Seed {seed} - Accuracy: {acc:.4f}, F1: {f1:.4f}")

        logging.info("\n✅ [Multi Run Result]")
        logging.info(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        logging.info(f"F1 Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    else:
        run_name = f"{model_name}_{str_target_classes}_{concat_patch_str}_seed42_{timestamp}"
        logging.info("Running single experiment...")
        acc, f1 = run_experiment(seed=42, cfg=cfg, model_name=model_name, final_output_dir=final_output_dir, device=device, tags=tags, run_name=run_name)
        logging.info(f"Single run - Accuracy: {acc:.4f}, F1: {f1:.4f}")


if __name__ == '__main__':
    main()

import os
import json
import logging
import torch
import torch.utils.data as torchUtils
import numpy as np
from datetime import datetime
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

import _init_path
from config import cfg, update_config
from dataset.joint_patches import FinalSamplesDataset
from utils.utils import EarlyStopping, BestModelSaver, stratified_split_dataset, check_label_distribution_from_subset
from core.patch_trainer import PatchTrainer
import models
import wandb


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

    dataset = FinalSamplesDataset(cfg)

    # train_size = int(0.8 * len(dataset))
    # val_size = int(0.1 * len(dataset))
    # test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = stratified_split_dataset(dataset, seed)

    check_label_distribution_from_subset(train_dataset, "Train")
    check_label_distribution_from_subset(val_dataset, "Validation")
    check_label_distribution_from_subset(test_dataset, "Test")

    logging.info(f"Train size: {len(train_dataset)}")
    logging.info(f"Val size: {len(val_dataset)}")
    logging.info(f"Test size: {len(test_dataset)}")

    train_loader = torchUtils.DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU, shuffle=True)
    val_loader = torchUtils.DataLoader(val_dataset, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU)
    test_loader = torchUtils.DataLoader(test_dataset, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU)

    model = eval('models.' + model_name + '.get_feature_extractor')(cfg, is_train=True).to(device)

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
        val_perf, val_loss, precision, recall, f1 = trainer.validate(epoch, model, val_loader)
        best_model_saver.save(model, val_loss)
        early_stopping(val_loss)
        run.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss,
                 "val_perf": val_perf, "precision": precision, "recall": recall, "f1": f1, "epoch": epoch})
        if early_stopping:
            logging.info("Early stopping triggered.")
            break

    model = best_model_saver.load_best_model(model).to(device).eval()
    test_perf, test_loss, precision, recall, f1 = trainer.validate(999, model, test_loader)

    run.log({"test_perf": test_perf, "test_loss": test_loss,
             "precision(test)": precision, "recall(test)": recall, "f1_score(test)": f1})
    run.finish()

    return test_perf, f1


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='config/large/tmp/origin_oa_normal.yaml', type=str)
    args = parser.parse_args()

    update_config(cfg, args)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_name = cfg.MODEL.NAME
    target_classes = cfg.DATASET.TARGET_CLASSES
    str_target_classes = '_'.join(target_classes)
    final_output_dir = os.path.join('output', f"multi_run_{timestamp}_{model_name}")
    logger = setup_logger(final_output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    extra = cfg.MODEL.EXTRA
    freeze = cfg.MODEL.FREEZE
    tags = [
        "foot-arthritis",
        f"TARGET={str_target_classes}",
        f"RAW={extra.RAW}",
        f"PATCH={extra.PATCH}",
        f"WITH_ATTN={extra.WITH_ATTN}",
        f"ONLYCAT={extra.ONLYCAT}",
        f"VIEWCAT={extra.VIEWCAT}",
        f"FREEZE_BACKBONE={freeze.BACKBONE}",
        f"FREEZE_CLASSIFIER={freeze.CLASSIFIER}",
        f"FREEZE_PROJECTION={freeze.PROJECTION}"
    ]

    if getattr(cfg.DATASET, 'MULTI_RUN', False):
        accs, f1s = [], []
        for seed in range(20):
            run_name = f"{model_name}_{str_target_classes}_seed{seed}_{timestamp}"
            logging.info(f"Running experiment {seed}...")
            acc, f1 = run_experiment(seed, cfg, model_name, final_output_dir, device, tags, run_name)
            accs.append(acc)
            f1s.append(f1)
            logging.info(f"Seed {seed} - Accuracy: {acc:.4f}, F1: {f1:.4f}")

        logging.info("\n✅ [Multi Run Result]")
        logging.info(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        logging.info(f"F1 Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    else:
        run_name = f"{model_name}_{str_target_classes}_seed42_{timestamp}"
        logging.info("Running single experiment...")
        acc, f1 = run_experiment(seed=42, cfg=cfg, model_name=model_name, final_output_dir=final_output_dir, device=device, tags=tags, run_name=run_name)
        logging.info(f"Single run - Accuracy: {acc:.4f}, F1: {f1:.4f}")


if __name__ == '__main__':
    main()

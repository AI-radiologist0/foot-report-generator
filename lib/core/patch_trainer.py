# -----------------------------------------------------------
# 
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
#
# -----------------------------------------------------------
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.utils import AverageMeter
from core.loss import FocalLoss
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc


import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import logging
import time

class PatchTrainer:
    def __init__(self, cfg, model, output_dir, writer_dict):
        self.device = 'cuda' if cfg.DEVICE == "GPU" else 'cpu'
        self.model = model.to(self.device)
        self.output_dir = output_dir
        self.print_freq = cfg.PRINT_FREQ
        self.writer_dict = writer_dict
        self.scheduler = cfg.TRAIN.SCHEDULER
        self.target_classes = cfg.DATASET.TARGET_CLASSES
        self.is_binary = len(self.target_classes) == 2

        # 🔹 Loss 자동 적용 (BCE vs Focal Loss)
        if cfg.TRAIN.LOSS == 'BCELoss':
            self.criterion = nn.BCELoss()
        elif cfg.TRAIN.LOSS == 'FocalLoss':
            self.criterion = FocalLoss(cfg)

    def train(self, epoch, data_loader, optimizer, scheduler=None):
        logger = logging.getLogger("Training")
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        correct, total = 0, 0
        criterion = self.criterion
        self.model.train()
        end = time.time()

        for i, (images, patches, labels) in enumerate(data_loader):
            data_time.update(time.time() - end)
            images, patches, labels = images.to(self.device), patches.to(self.device), labels.to(self.device)

            # 🔹 이진 분류(BCE) vs 다중 분류(CE) 적용
            if self.is_binary:
                labels = labels.float() # BCE Loss 적용을 위해 차원 확장
                outputs = self.model(images, patches)
            else:
                labels = labels.long()
                outputs = self.model(images, patches)  # 다중 분류에서는 Softmax 미적용 (CrossEntropy Loss가 내부적으로 적용)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            # 🔹 정확도 계산 (이진 분류 & 다중 분류)
            if self.is_binary:
                preds = (outputs > 0.5).float()  # BCE에서는 0.5 기준으로 분류
            else:
                preds = torch.argmax(outputs, dim=1)  # 다중 분류에서는 argmax 사용

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # 로그 출력
            if i % self.print_freq == 0:
                msg = f'Epoch: [{epoch}] [{i}/{len(data_loader)}] \t ' \
                      f'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s) \t' \
                      f'Data Time: {data_time.val:.3f}s ({data_time.avg:.3f}s) \t' \
                      f'Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f})'
                logger.info(msg)

            writer = self.writer_dict['writer']
            global_steps = self.writer_dict['train_global_steps']
            writer.add_scalar('train_loss', loss_meter.val, global_steps)
            self.writer_dict['train_global_steps'] = global_steps + 1

        train_acc = correct / total
        acc_meter.update(correct, total)
        writer.add_scalar('train_accuracy', acc_meter.avg, global_steps - 1)
        logger.info(f'Epoch [{epoch}] - Training Acc : {train_acc:.4f} ')

        return loss_meter.avg, train_acc

    def validate(self, epoch, model, val_loader, writer_dict=None, criterion=None):
        logger = logging.getLogger("Validation")
        model.eval()

        if not criterion:
            criterion = self.criterion

        losses = AverageMeter()
        accuracies = AverageMeter()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, patches, labels in tqdm(val_loader):
                images, patches, labels = images.to(self.device), patches.to(self.device), labels.to(self.device)

                # 🔹 이진 분류(BCE) vs 다중 분류(CE) 적용
                if self.is_binary:
                    labels = labels.float()  # BCE Loss 적용을 위해 차원 확장
                    outputs = model(images, patches)  # Sigmoid 적용
                else:
                    labels = labels.long()
                    outputs = model(images, patches)  # Softmax 미적용 (CrossEntropy Loss 내부적으로 적용)

                loss = criterion(outputs, labels)
                losses.update(loss.item(), images.size(0))

                # 🔹 정확도 계산 (이진 분류 & 다중 분류)
                if self.is_binary:
                    preds = (outputs > 0.5).float()  # BCE에서는 0.5 기준으로 분류
                else:
                    preds = torch.argmax(outputs, dim=1)  # 다중 분류에서는 argmax 사용

                correct = (preds == labels).sum().item()
                total = labels.size(0)
                accuracy = correct / total
                accuracies.update(accuracy, total)

                all_probs.extend(outputs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 🔹 Precision, Recall, F1 Score 계산
        if self.is_binary:
            precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        else:
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        # TensorBoard 기록
        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_acc', accuracies.avg * 100, global_steps)
            writer.add_scalar('valid_precision', precision, global_steps)
            writer.add_scalar('valid_recall', recall, global_steps)
            writer.add_scalar('valid_f1', f1, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

        # 로그 출력
        logger.info(f'[Epoch {epoch}] Validation Accuracy: {accuracies.avg * 100:.2f}% \t Loss: {losses.avg:.4f}')
        logger.info(f'[Epoch {epoch}] Precision: {precision:.4f} \t Recall: {recall:.4f} \t F1 Score: {f1:.4f}')

        return accuracies.avg * 100, losses.avg, precision, recall, f1

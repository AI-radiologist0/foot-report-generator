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
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc


class PatchTrainer:
    def __init__(self, cfg, model, output_dir, writer_dict):
        self.device = 'cuda' if cfg.DEVICE == "GPU" else 'cpu'
        self.model = model
        self.output_dir = output_dir  # cfg로 설정
        self.print_freq = cfg.PRINT_FREQ
        self.writer_dict = writer_dict
        self.val_loss = None
        self.val_accuracy = None
        self.scheduler = cfg.TRAIN.SCHEDULER
        self.criterion = nn.BCELoss()


    def train(self, epoch, data_loader, optimizer, scheduler=None):
        # logger
        logger = logging.getLogger("Training")
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        correct, total = 0, 0
        
        criterion = nn.BCELoss()
        self.model.train()

        end = time.time()

        for i, (images, patches, labels )in enumerate(data_loader):

            data_time.update(time.time() - end)
            images, patches, labels = images.to(self.device), patches.to(self.device), labels.float().to(self.device).unsqueeze(1)
          
            outputs = self.model(images, patches)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_meter.update(loss.item(), images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            
            correct += ((outputs > 0.5).float() == labels).sum().item()
            total += labels.size(0)

            # logger -----)
            if i % self.print_freq == 0:
                msg = f'Epoch: [{epoch}] [{i}/{len(data_loader)}] \t ' \
                      f'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s) \t' \
                      f'Data Time: {data_time.val:.3f}s ({data_time.avg:.3f}s) \t' \
                      f'Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f})'
                logger.info(msg)

                # logger.info(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

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
        model.eval()  # 모델을 평가 모드로 설정

        if not criterion:
            criterion = self.criterion


        losses = AverageMeter()  # 손실 추적
        accuracies = AverageMeter()  # 정확도 추적

        all_preds = []
        all_labels = []
        all_probs = []  # 확률값 저장 (ROC 곡선 용도)

        with torch.no_grad():
            end = time.time()
            for images, patches, labels in tqdm(val_loader):
                # 데이터 로드 및 디바이스로 이동
                images, patches, labels = images.to(self.device), patches.to(self.device), labels.float().to(self.device).unsqueeze(1)

                # 모델 예측
                outputs = model(images, patches)
                loss = criterion(outputs, labels)

                # 손실 업데이트
                losses.update(loss.item(), images.size(0))

                preds = (outputs > 0.5).float()

                # 정확도 계산 및 업데이트
                correct = (preds == labels).sum().item()
                total = labels.size(0)
                accuracy = correct / total
                accuracies.update(accuracy, total)

                # 예측 및 실제 값 저장
                all_probs.extend(outputs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Precision, Recall, F1 Score 계산 (이진 분류)
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

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

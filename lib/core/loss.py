# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch
import torch.nn.functional as F


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


# ===========================================================
class FocalLoss(nn.Module):
    def __init__(self, cfg, alpha=None, gamma=2, reduction='mean'):
        """
        Multi-Class Focal Loss Implementation.
        
        Args:
            alpha (Tensor, list, float, optional): Class weighting factor.
                - If `None`, no class balancing is applied.
                - If `float`, the same alpha is applied to all classes.
                - If `list` or `Tensor`, should be of shape [num_classes].
            gamma (float): Focusing parameter.
            reduction (str): 'mean' (default), 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.target_classes = cfg.DATASET.TARGET_CLASSES
        
        if alpha is not None:
            if isinstance(alpha, (float, int)):  # 단일 값이 들어오면 모든 클래스에 동일하게 적용
                self.alpha = torch.tensor([alpha] * len(self.target_classes))
            elif isinstance(alpha, (list, torch.Tensor)):  # 클래스별 가중치 적용
                self.alpha = torch.tensor(alpha)
            else:
                raise TypeError("alpha must be float, list, or torch.Tensor")
        else:
            self.alpha = None  # 클래스별 가중치 없음

    def forward(self, inputs, targets):
        """
        Compute Focal Loss for multi-class classification.

        Args:
            inputs (Tensor): Logits from the model (shape: [batch_size, num_classes]).
            targets (Tensor): Ground truth labels (shape: [batch_size], values in [0, num_classes-1]).

        Returns:
            Tensor: Computed Focal Loss.
        """
        num_classes = inputs.shape[1]
        log_probs = F.log_softmax(inputs, dim=1)  # Log-probabilities
        probs = torch.exp(log_probs)  # Convert log-probs to probs

        # One-hot encoding of targets
        targets_one_hot = F.one_hot(targets.squeeze().to(torch.long), num_classes=num_classes).float()  # Shape: [batch_size, num_classes]
        pt = (probs * targets_one_hot).sum(dim=1)  # Select p_t for target class

        # Compute focal weight (1 - p_t)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # Compute log loss for target class
        log_loss = (-log_probs * targets_one_hot).sum(dim=1)  # Cross-entropy loss per sample

        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha_t = (self.alpha.to(inputs.device) * targets_one_hot).sum(dim=1)
            focal_loss = alpha_t * focal_weight * log_loss
        else:
            focal_loss = focal_weight * log_loss

        # Apply reduction method
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss  # 'none' case
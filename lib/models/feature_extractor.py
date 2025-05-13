import logging
import torch
import torch.nn as nn
import timm
from torchvision import models
import torch.nn.functional as F

def create_swin_model(variant, pretrained):
    model = timm.create_model(variant, pretrained=pretrained)
    return model, model.num_features

def create_resnet_model(pretrained, in_channels=3):
    model = models.resnet50(pretrained=pretrained)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, dim

def get_model(cfg, pretrained=True):
    raw_type = cfg.MODEL.EXTRA.RAW.lower()
    patch_type = cfg.MODEL.EXTRA.PATCH.lower()

    if raw_type == 'swin-t':
        global_model, global_dim = create_swin_model('swin_tiny_patch4_window7_224', pretrained)
    elif raw_type == 'resnet':
        global_model, global_dim = create_resnet_model(pretrained, in_channels=3)
    else:
        raise ValueError(f"Unsupported RAW model type: {raw_type}")

    if patch_type == 'swin-t':
        local_model, local_dim = create_swin_model('swin_tiny_patch4_window7_224', pretrained)
    elif patch_type == 'resnet':
        local_model, local_dim = create_resnet_model(pretrained, in_channels=3)
    else:
        raise ValueError(f"Unsupported PATCH model type: {patch_type}")

    return global_dim, global_model, local_dim, local_model

class FeatureExtractor(nn.Module):
    def __init__(self, cfg, pretrained=True, return_sequence=False, return_output_vector=False, **kwarg):
        super(FeatureExtractor, self).__init__()

        self.target_classes = cfg.DATASET.TARGET_CLASSES
        self.is_binary = len(self.target_classes) == 2
        self.output_dim = 1 if self.is_binary else len(self.target_classes)
        self.return_sequence = return_sequence
        self.return_output_vector = return_output_vector

        self.global_feature_dim, self.global_branch, self.local_feature_dim, self.local_branch = get_model(cfg, pretrained=pretrained)
        self.patch_proj = nn.Linear(self.local_feature_dim, self.global_feature_dim)

        assert isinstance(self.global_feature_dim, int)
        assert isinstance(self.local_feature_dim, int)
        assert isinstance(self.global_branch, nn.Module)
        assert isinstance(self.local_branch, nn.Module)

        logging.info(f"local: {self.local_feature_dim}, global: {self.global_feature_dim}")

        self.classifier = nn.Sequential(
            nn.Linear(self.global_feature_dim + self.local_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)
        )

        if cfg.MODEL.EXTRA.USE_CKPT and cfg.MODEL.EXTRA.CKPT:
            logging.info(f"feature extractor load ckpt flag: {cfg.MODEL.EXTRA.USE_CKPT}")
            self.load_from_checkpoint(cfg.MODEL.EXTRA.CKPT)

    def forward(self, image, patches):
        global_features = self.global_branch.forward_features(image)
        if global_features.dim() == 4:
            global_features = global_features.mean(dim=[1, 2])
        elif global_features.dim() == 3:
            global_features = global_features.mean(dim=1)
        global_features = global_features.unsqueeze(1)

        B, N, C, H, W = patches.shape
        patches = patches.view(B * N, C, H, W)
        local_features = self.local_branch(patches)
        local_features = self.patch_proj(local_features)
        local_features = local_features.view(B, N, -1)

        image_tokens = torch.cat([global_features, local_features], dim=1)  # (B, 1+N, C)

        if self.return_sequence:
            return image_tokens  # for Transformer input

        global_vector = global_features.squeeze(1)
        local_vector = local_features.mean(dim=1)
        combined_features = torch.cat((global_vector, local_vector), dim=1)

        if self.return_output_vector:
            return combined_features  # for GPT-style decoder input

        output = self.classifier(combined_features)
        return torch.sigmoid(output) if self.is_binary else output

    def load_from_checkpoint(self, checkpoint_path, map_location=None):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.load_state_dict(checkpoint)
        logging.info(f"Model weights loaded from {checkpoint_path}")

def get_feature_extractor(cfg, is_train, remove_classifier=False, **kwargs):
    model = FeatureExtractor(cfg, **kwargs)
    if remove_classifier and hasattr(model, 'classifier'):
        model.classifier = None
    model.train() if is_train else model.eval()
    return model

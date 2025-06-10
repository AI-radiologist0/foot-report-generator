import logging
import torch
import torch.nn as nn
import timm
from torchvision import models
import torch.nn.functional as F

logger = logging.getLogger("FreezeLogger")
logger.setLevel(logging.INFO)

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

    def load_model(model_type: str, in_channels: int = 3):
        is_swin = 'swin' in model_type
        if is_swin:
            model = timm.create_model(model_type, pretrained=pretrained)
            model.head = nn.Identity()
            dim = model.num_features
        elif model_type == 'resnet':
            model = models.resnet50(pretrained=pretrained)
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            dim = model.fc.in_features
            model.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        return dim, model, is_swin

    # Global
    global_model_type = 'swin_tiny_patch4_window7_224' if raw_type == 'swin-t' else 'resnet'
    global_dim, global_model, global_is_swin = load_model(global_model_type)

    # Local
    patch_model_type = 'swin_tiny_patch4_window7_224' if patch_type == 'swin-t' else 'resnet'
    in_channels = 3 if 'swin' in patch_model_type else 102
    local_dim, local_model, local_is_swin = load_model(patch_model_type, in_channels)

    return global_dim, global_model, global_is_swin, local_dim, local_model, local_is_swin


def pooled_swin_features(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        if x.shape[1] == x.shape[2] == 7:  # [B, 7, 7, 768]
            return x.mean(dim=[1, 2])
        elif x.shape[2] == x.shape[3] == 7:  # [B, 768, 7, 7]
            return x.mean(dim=[2, 3])
    raise ValueError(f"Unexpected Swin output shape: {x.shape}")


class FeatureExtractorV3(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.cfg = cfg

        self.global_feature_dim, self.global_branch, self.global_is_swin, \
        self.local_feature_dim, self.local_branch, self.local_is_swin = get_model(cfg, pretrained)

        self.output_dim = 1 if len(cfg.DATASET.TARGET_CLASSES) == 2 else len(cfg.DATASET.TARGET_CLASSES)
        self.proj_dim = 1024  # 공통 투영 차원

        self.global_proj = nn.Sequential(
            nn.Linear(self.global_feature_dim, self.proj_dim),
            nn.BatchNorm1d(self.proj_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.local_proj = nn.Sequential(
            nn.Linear(self.local_feature_dim, self.proj_dim),
            nn.BatchNorm1d(self.proj_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.proj_dim * 2, self.output_dim),
            nn.Sigmoid() if self.output_dim == 1 else nn.Identity()
        )

        self.patch_resize = nn.Conv2d(102, 3, kernel_size=1)

    def forward(self, image: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        # Global
        global_feat = self.global_branch.forward_features(image) if self.global_is_swin else self.global_branch(image)
        global_feat = pooled_swin_features(global_feat) if self.global_is_swin else global_feat
        global_feat = self.global_proj(global_feat)

        # Local
        B, N, C, H, W = patches.shape
        patches = patches.view(B, N * C, H, W)
        patches = self.patch_resize(patches) if self.local_is_swin else patches
        patches = F.interpolate(patches, size=(224, 224), mode='bilinear', align_corners=False) if self.local_is_swin else patches
        local_feat = self.local_branch.forward_features(patches) if self.local_is_swin else self.local_branch(patches)
        local_feat = pooled_swin_features(local_feat) if self.local_is_swin else local_feat
        local_feat = self.local_proj(local_feat)

        # Concat and classify
        combined = torch.cat([global_feat, local_feat], dim=1)
        out = self.classifier(combined)
        return out


def freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False

def apply_freeze(model: nn.Module, cfg):
    if getattr(cfg.MODEL.FREEZE, 'BACKBONE', False):
        freeze_module(model.global_branch)
        freeze_module(model.local_branch)

    if getattr(cfg.MODEL.FREEZE, 'PROJECTION', False):
        if hasattr(model, 'patch_proj'):
            freeze_module(model.patch_proj)

    if getattr(cfg.MODEL.FREEZE, 'CLASSIFIER', False):
        if hasattr(model, 'classifier') and model.classifier is not None:
            freeze_module(model.classifier)

def log_freeze_status(model: nn.Module, logger: logging.Logger, name: str = ""):
    logger.info(f"🔍 [Freeze Status] {name}")
    for param_name, param in model.named_parameters():
        status = "🔒 FROZEN" if not param.requires_grad else "✅ TRAINABLE"
        logger.info(f"{status:12} | {name}.{param_name}")

def get_feature_extractor(cfg, is_train, remove_classifier=False, **kwargs):
    model = FeatureExtractorV3(cfg, **kwargs)
    
    if remove_classifier and hasattr(model, 'classifier'):
        model.classifier = None
    
    apply_freeze(model, cfg)

    # logging
    logger = logging.getLogger("FreezeLogger")
    logger.setLevel(logging.INFO)
    log_freeze_status(model.global_branch, logger, "Global Branch")
    log_freeze_status(model.local_branch, logger, "Local Branch")
    if hasattr(model, 'patch_proj'):
        log_freeze_status(model.patch_proj, logger, "Patch Projection")
    if hasattr(model, 'classifier') and model.classifier is not None:
        log_freeze_status(model.classifier, logger, "Classifier")

    model.train() if is_train else model.eval()
    return model

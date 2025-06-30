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
    raw_type = cfg.MODEL.EXTRA.RAW.lower() if cfg.MODEL.EXTRA.RAW is not None else None
    patch_type = cfg.MODEL.EXTRA.PATCH.lower() if cfg.MODEL.EXTRA.PATCH is not None else None


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
        elif model_type == "none":
            return 0, None, False
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        return dim, model, is_swin

    # Global
    global_model_type = 'swin_tiny_patch4_window7_224' if raw_type == 'swin-t' else 'resnet'
    global_dim, global_model, global_is_swin = load_model(global_model_type)

    # Local
    if patch_type is None or patch_type == 'none':
        local_dim, local_model, local_is_swin = load_model("none")
    else:
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

        self.local_proj = None
        if self.local_branch is not None:
            self.local_proj = nn.Sequential(
                nn.Linear(self.local_feature_dim, self.proj_dim),
                nn.BatchNorm1d(self.proj_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            )

        self.classifier = None
        if self.global_branch is not None and self.local_branch is not None:
            self.classifier = nn.Sequential(
                nn.Linear(self.proj_dim * 2, self.output_dim),
                nn.Sigmoid() if self.output_dim == 1 else nn.Identity()
            )
        elif self.global_branch is not None:
            self.classifier = nn.Sequential(
                nn.Linear(self.proj_dim, self.output_dim),
                nn.Sigmoid() if self.output_dim == 1 else nn.Identity()
            )
        elif self.local_branch is not None:
            self.classifier = nn.Sequential(
                nn.Linear(self.proj_dim, self.output_dim),
                nn.Sigmoid() if self.output_dim == 1 else nn.Identity()
            )

        self.patch_resize = None
        if self.local_branch is not None:
            self.patch_resize = nn.Conv2d(102, 3, kernel_size=1)

    def forward(self, image: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        # Global
        if self.global_branch is not None:
            global_feat = self.global_branch.forward_features(image) if self.global_is_swin else self.global_branch(image)
            global_feat = pooled_swin_features(global_feat) if self.global_is_swin else global_feat
            global_feat = self.global_proj(global_feat)

        else:
            global_feat = None

        # Local
        B, N, C, H, W = patches.shape
        patches = patches.view(B, N * C, H, W)
        if self.patch_resize is not None:
            patches = self.patch_resize(patches) if self.local_is_swin else patches
            patches = F.interpolate(patches, size=(224, 224), mode='bilinear', align_corners=False) if self.local_is_swin else patches
        if self.local_branch is not None:
            local_feat = self.local_branch.forward_features(patches) if self.local_is_swin else self.local_branch(patches)
            local_feat = pooled_swin_features(local_feat) if self.local_is_swin else local_feat
            local_feat = self.local_proj(local_feat)
        else:
            local_feat = None

        # Concat and classify
        if global_feat is not None and local_feat is not None:
            combined = torch.cat([global_feat, local_feat], dim=1)
            out = self.classifier(combined)
        elif global_feat is not None:
            out = self.classifier(global_feat)
        elif local_feat is not None:
            out = self.classifier(local_feat)
        else:
            raise ValueError("No features to classify")
        return out


def freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False

def apply_freeze(model: nn.Module, cfg):
    if getattr(cfg.MODEL.FREEZE, 'BACKBONE', False):
        if model.global_branch is not None:
            freeze_module(model.global_branch)
        if model.local_branch is not None:
            freeze_module(model.local_branch)

    if getattr(cfg.MODEL.FREEZE, 'PROJECTION', False):
        if hasattr(model, 'global_proj') and model.global_proj is not None:
            freeze_module(model.global_proj)
        if hasattr(model, 'local_proj') and model.local_proj is not None:
            freeze_module(model.local_proj)

    if getattr(cfg.MODEL.FREEZE, 'CLASSIFIER', False):
        if hasattr(model, 'classifier') and model.classifier is not None:
            freeze_module(model.classifier)

def log_freeze_status(model: nn.Module, logger: logging.Logger, name: str = ""):
    logger.info(f"🔍 [Freeze Status] {name}")
    if model is None:
        logger.info(f"❌ {name} is None (not used)")
        return
    
    for param_name, param in model.named_parameters():
        status = "🔒 FROZEN" if not param.requires_grad else "✅ TRAINABLE"
        logger.info(f"{status:12} | {name}.{param_name}")

def log_model_configuration(cfg, logger: logging.Logger):
    """모델 구성 상태를 로깅"""
    logger.info("=" * 60)
    logger.info("🏗️  [Model Configuration]")
    logger.info("=" * 60)
    
    raw_type = cfg.MODEL.EXTRA.RAW
    patch_type = cfg.MODEL.EXTRA.PATCH
    
    logger.info(f"📋 RAW (Global) Model: {raw_type}")
    logger.info(f"📋 PATCH (Local) Model: {patch_type}")
    
    # 모델 활성화 상태 확인
    global_active = raw_type is not None and raw_type.lower() != "none"
    local_active = patch_type is not None and patch_type.lower() != "none"
    
    logger.info(f"🌐 Global Branch: {'✅ ACTIVE' if global_active else '❌ INACTIVE'}")
    logger.info(f"🔍 Local Branch: {'✅ ACTIVE' if local_active else '❌ INACTIVE'}")
    
    if global_active and local_active:
        logger.info("🎯 Mode: Dual Branch (Global + Local)")
    elif global_active:
        logger.info("🎯 Mode: Global Only")
    elif local_active:
        logger.info("🎯 Mode: Local Only")
    else:
        logger.info("❌ ERROR: No active branches!")
    
    logger.info("=" * 60)

def log_model_parameters(model: nn.Module, logger: logging.Logger):
    """모델 파라미터 통계를 로깅"""
    logger.info("📊 [Model Parameters Statistics]")
    logger.info("=" * 60)
    
    total_params = 0
    trainable_params = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight') or hasattr(module, 'bias'):
            module_params = sum(p.numel() for p in module.parameters())
            module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            if module_params > 0:
                total_params += module_params
                trainable_params += module_trainable
                logger.info(f"{name:30} | Params: {module_params:>8,} | Trainable: {module_trainable:>8,}")
    
    logger.info("-" * 60)
    logger.info(f"{'TOTAL':30} | Params: {total_params:>8,} | Trainable: {trainable_params:>8,}")
    logger.info(f"Frozen params: {total_params - trainable_params:,}")
    logger.info("=" * 60)

def log_branch_details(model: nn.Module, logger: logging.Logger):
    """각 브랜치의 상세 정보를 로깅"""
    logger.info("🔧 [Branch Details]")
    logger.info("=" * 60)
    
    # Global Branch
    if model.global_branch is not None:
        logger.info("🌐 Global Branch:")
        logger.info(f"   - Type: {'Swin Transformer' if model.global_is_swin else 'ResNet'}")
        logger.info(f"   - Feature Dim: {model.global_feature_dim}")
        logger.info(f"   - Projection: {model.global_proj}")
    else:
        logger.info("🌐 Global Branch: ❌ None")
    
    # Local Branch
    if model.local_branch is not None:
        logger.info("🔍 Local Branch:")
        logger.info(f"   - Type: {'Swin Transformer' if model.local_is_swin else 'ResNet'}")
        logger.info(f"   - Feature Dim: {model.local_feature_dim}")
        logger.info(f"   - Projection: {model.local_proj}")
    else:
        logger.info("🔍 Local Branch: ❌ None")
    
    # Classifier
    if model.classifier is not None:
        logger.info("🎯 Classifier:")
        logger.info(f"   - Output Dim: {model.output_dim}")
        logger.info(f"   - Architecture: {model.classifier}")
    else:
        logger.info("🎯 Classifier: ❌ None")
    
    logger.info("=" * 60)

def get_feature_extractor(cfg, is_train, remove_classifier=False, **kwargs):
    model = FeatureExtractorV3(cfg, **kwargs)
    
    if remove_classifier and hasattr(model, 'classifier'):
        model.classifier = None
    
    apply_freeze(model, cfg)

    # Enhanced logging
    logger = logging.getLogger("FreezeLogger")
    logger.setLevel(logging.INFO)
    
    # 1. 모델 구성 상태 로깅
    log_model_configuration(cfg, logger)
    
    # 2. 브랜치 상세 정보 로깅
    log_branch_details(model, logger)
    
    # 3. Freeze 상태 로깅
    logger.info("🔒 [Freeze Status Details]")
    logger.info("=" * 60)
    log_freeze_status(model.global_branch, logger, "Global Branch")
    log_freeze_status(model.local_branch, logger, "Local Branch")
    if hasattr(model, 'global_proj') and model.global_proj is not None:
        log_freeze_status(model.global_proj, logger, "Global Projection")
    if hasattr(model, 'local_proj') and model.local_proj is not None:
        log_freeze_status(model.local_proj, logger, "Local Projection")
    if hasattr(model, 'classifier') and model.classifier is not None:
        log_freeze_status(model.classifier, logger, "Classifier")
    
    # 4. 파라미터 통계 로깅
    log_model_parameters(model, logger)

    model.train() if is_train else model.eval()
    return model

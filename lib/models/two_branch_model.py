import torch
import torch.nn as nn
import timm
from torchvision import models
from .utils import freeze_module, apply_freeze, log_freeze_status, log_model_parameters, log_model_configuration, log_branch_details

def pooled_swin_features(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        if x.shape[1] == x.shape[2] == 7:  # [B, 7, 7, 768]
            return x.mean(dim=[1, 2])
        elif x.shape[2] == x.shape[3] == 7:  # [B, 768, 7, 7]
            return x.mean(dim=[2, 3])
    raise ValueError(f"Unexpected Swin output shape: {x.shape}")

class FeatureExtractor(nn.Module):
    def __init__(self, backbone='resnet', pretrained=True):
        super().__init__()
        if backbone == 'resnet':
            model = models.resnet50(pretrained=pretrained)
            self.feature_dim = model.fc.in_features
            self.backbone = nn.Sequential(*list(model.children())[:-1])  # fc 제외
        elif backbone == 'swin-t':
            model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained)
            self.feature_dim = model.num_features
            self.backbone = model
        else:
            raise ValueError('Unsupported backbone')

    def forward(self, x):
        if isinstance(self.backbone, nn.Sequential):
            # ResNet
            x = self.backbone(x)  # (B, feature_dim, 1, 1)
            x = torch.flatten(x, 1)
        else:
            # Swin-T
            x = self.backbone.forward_features(x)
            x = pooled_swin_features(x)
        return x

class LocalExtractor(nn.Module):
    def __init__(self, in_patches=34, out_channels=3):
        super().__init__()
        self.conv = nn.Conv2d(in_patches, out_channels, kernel_size=1)

    def forward(self, patches):
        # patches: (B, 34, 1, 112, 112)
        B, N, C, H, W = patches.shape
        x = patches.view(B, N, H, W)  # (B, 34, 112, 112)
        x = self.conv(x)              # (B, 3, 112, 112)
        return x

def get_model(cfg, pretrained=True):
    raw_type = cfg.MODEL.EXTRA.RAW.lower() if cfg.MODEL.EXTRA.RAW is not None else None
    patch_type = cfg.MODEL.EXTRA.PATCH.lower() if cfg.MODEL.EXTRA.PATCH is not None else None

    # Global branch
    if raw_type == 'swin-t':
        global_extractor = FeatureExtractor(backbone='swin-t', pretrained=pretrained)
    else:
        global_extractor = FeatureExtractor(backbone='resnet', pretrained=pretrained)
    global_dim = global_extractor.feature_dim
    global_is_swin = (raw_type == 'swin-t')

    # Local branch
    local_patch_conv = LocalExtractor(in_patches=34, out_channels=3)
    if patch_type == 'swin-t':
        local_extractor = FeatureExtractor(backbone='swin-t', pretrained=pretrained)
    else:
        local_extractor = FeatureExtractor(backbone='resnet', pretrained=pretrained)
    local_dim = local_extractor.feature_dim
    local_is_swin = (patch_type == 'swin-t')

    return global_dim, global_extractor, global_is_swin, local_dim, local_patch_conv, local_extractor, local_is_swin

class TwoBranchModel(nn.Module):
    def __init__(self, cfg, pretrained=True, proj_dim=1024, num_classes=2):
        super().__init__()
        # get_model에서 모든 branch/conv를 받아옴
        gdim, self.global_extractor, self.global_is_swin, ldim, self.local_patch_conv, self.local_extractor, self.local_is_swin = get_model(cfg, pretrained)
        self.proj_dim = proj_dim
        self.output_dim = 1 if num_classes == 2 else num_classes
        self.raw_proj = nn.Sequential(
            nn.Linear(gdim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.patch_proj = nn.Sequential(
            nn.Linear(ldim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        # classifier: 항상 로짓만 출력
        self.classifier = nn.Linear(proj_dim * 2, self.output_dim)
        self.concat_patches = cfg.DATASET.CONCAT_PATCH

    def forward(self, image, patches):
        # Raw branch
        raw_feat = self.global_extractor(image)
        raw_feat = self.raw_proj(raw_feat)
        # Patch branch
        if self.concat_patches:
            # patches: (B, 1, 224, 224) → 바로 local_extractor에 입력
            patch_img = patches
        else:
            # patches: (B, 34, 1, 112, 112) → LocalExtractor 거침
            patch_img = self.local_patch_conv(patches)
        patch_feat = self.local_extractor(patch_img)
        patch_feat = self.patch_proj(patch_feat)
        # Concat
        feat = torch.cat([raw_feat, patch_feat], dim=1)
        out = self.classifier(feat)
        return out

    def log_model_info(self, cfg, logger):
        log_model_configuration(cfg, logger)
        log_branch_details(self, logger)
        logger.info("🔒 [Freeze Status Details]")
        logger.info("=" * 60)
        if hasattr(self, 'global_extractor') and self.global_extractor is not None:
            log_freeze_status(self.global_extractor, logger, "Global Extractor")
        if hasattr(self, 'local_extractor') and self.local_extractor is not None:
            log_freeze_status(self.local_extractor, logger, "Local Extractor")
        if hasattr(self, 'local_patch_conv') and self.local_patch_conv is not None:
            log_freeze_status(self.local_patch_conv, logger, "Local Patch Conv")
        if hasattr(self, 'raw_proj') and self.raw_proj is not None:
            log_freeze_status(self.raw_proj, logger, "Raw Projection")
        if hasattr(self, 'patch_proj') and self.patch_proj is not None:
            log_freeze_status(self.patch_proj, logger, "Patch Projection")
        if hasattr(self, 'classifier') and self.classifier is not None:
            log_freeze_status(self.classifier, logger, "Classifier")
        log_model_parameters(self, logger)

def get_feature_extractor(cfg, is_train=True, remove_classifier=False, **kwargs):
    num_classes = 1 if len(cfg.DATASET.TARGET_CLASSES) == 2 else len(cfg.DATASET.TARGET_CLASSES)
    model = TwoBranchModel(cfg, pretrained=True, proj_dim=1024, num_classes=num_classes)
    if remove_classifier:
        model.classifier = nn.Sequential()
    model.train() if is_train else model.eval()
    return model 
import logging
import torch
import torch.nn as nn
import timm
from torchvision import models
import torch.nn.functional as F

logger = logging.getLogger("FreezeLogger")
logger.setLevel(logging.INFO)

def pooled_swin_features(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        if x.shape[1] == x.shape[2] == 7:  # [B, 7, 7, 768]
            return x.mean(dim=[1, 2])
        elif x.shape[2] == x.shape[3] == 7:  # [B, 768, 7, 7]
            return x.mean(dim=[2, 3])
    raise ValueError(f"Unexpected Swin output shape: {x.shape}")


class BaseTwoBranchModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.swin_global = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained)
        self.swin_global.head = nn.Identity()
        self.swin_patch = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained)
        self.swin_patch.head = nn.Identity()
        self.swin_dim = self.swin_global.num_features

        resnet_global = models.resnet50(pretrained=pretrained)
        resnet_patch = models.resnet50(pretrained=pretrained)
        self.resnet_dim = resnet_global.fc.in_features
        resnet_global.fc = nn.Identity()
        resnet_patch.fc = nn.Identity()
        self.resnet_global = resnet_global
        self.resnet_patch = resnet_patch


class FeatureAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 4),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Linear(feature_dim, 512)

    def forward(self, feature_list):
        combined = torch.cat(feature_list, dim=1)
        weights = self.attention(combined)
        weighted_features = torch.stack(feature_list, dim=1) * weights.unsqueeze(-1)
        weighted_features = weighted_features.sum(dim=1)
        return self.fc(weighted_features)


class TwoBranchModelWithAttn(BaseTwoBranchModel):
    def __init__(self, pretrained=True):
        super().__init__(pretrained)
        self.patch_channel_reduction = nn.Conv2d(in_channels=102, out_channels=3, kernel_size=1)
        self.feature_attention = FeatureAttention(self.resnet_dim)
        self.swin_global_fc = nn.Linear(self.swin_dim, self.resnet_dim)
        self.swin_patch_fc = nn.Linear(self.swin_dim, self.resnet_dim)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, image, patches):
        B, N, C, H, W = patches.shape
        patches_reshaped = patches.view(B, N * C, H, W)
        patches_resized = F.interpolate(patches_reshaped, size=(224, 224), mode='bilinear', align_corners=False)
        patches_reduced = self.patch_channel_reduction(patches_resized)

        swin_global_features = pooled_swin_features(self.swin_global.forward_features(image))
        swin_patch_features = pooled_swin_features(self.swin_patch.forward_features(patches_reduced))

        resnet_global_features = self.resnet_global(image)
        resnet_patch_features = self.resnet_patch(patches_reduced).view(B, -1)

        swin_global_features = self.swin_global_fc(swin_global_features)
        swin_patch_features = self.swin_patch_fc(swin_patch_features)

        combined_features = self.feature_attention([
            swin_global_features, swin_patch_features,
            resnet_global_features, resnet_patch_features
        ])
        return self.classifier(combined_features)


class TwoBranchModelOnlyCat(BaseTwoBranchModel):
    def __init__(self, pretrained=True):
        super().__init__(pretrained)
        self.resnet_patch.conv1 = nn.Conv2d(102, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.patch_resize = nn.Conv2d(102, 3, kernel_size=1)
        self.classifier = nn.Sequential(nn.Linear(1, 1))

    def forward(self, image: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = patches.shape  # ex) [B, 34, 3, 112, 112]

        # 채널 차원으로 결합: [B, N*C, H, W] → [B, 102, 112, 112]
        patch_cat = patches.view(B, N * C, H, W)

        # 🔁 Swin 용도: 채널 3으로 변환 → 224x224 resize
        patches_for_swin = self.patch_resize(patch_cat)  # [B, 3, H, W]
        patches_for_swin = F.interpolate(patches_for_swin, size=(224, 224), mode='bilinear', align_corners=False)

        # 🔁 ResNet 용도: 102채널 그대로 사용224x224 resize
        patches_for_resnet = F.interpolate(patch_cat, size=(224, 224), mode='bilinear', align_corners=False)

        # Feature 추출
        swin_global_features = pooled_swin_features(self.swin_global.forward_features(image))
        swin_patch_features = pooled_swin_features(self.swin_patch.forward_features(patches_for_swin))

        resnet_global_features = self.resnet_global(image)
        resnet_patch_features = self.resnet_patch(patches_for_resnet)

        # 모든 feature를 연결
        combined_features = torch.cat([
            swin_global_features, swin_patch_features,
            resnet_global_features, resnet_patch_features
        ], dim=1)

        # classifier input 크기와 맞지 않으면 동적으로 생성
        if combined_features.shape[1] != self.classifier[0].in_features:
            self.classifier = nn.Sequential(
                nn.Linear(combined_features.shape[1], 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 1),
                nn.Sigmoid()
            ).to(image.device)

        return self.classifier(combined_features)


class TwoBranchModelViewCat(BaseTwoBranchModel):
    def __init__(self, pretrained=True):
        super().__init__(pretrained)
        self.res_proj_global = nn.Linear(self.resnet_dim, self.swin_dim)
        self.res_proj_patch = nn.Linear(self.resnet_dim, self.swin_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.swin_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, image, patches):
        B, N, C, H, W = patches.shape
        patch_batch = patches.view(B * N, C, H, W)

        swin_global = pooled_swin_features(self.swin_global.forward_features(image)).unsqueeze(1)
        swin_patch = self.swin_patch.forward_features(F.interpolate(patch_batch, size=(224, 224), mode='bilinear', align_corners=False))
        swin_patch = pooled_swin_features(swin_patch).view(B, N, -1)

        resnet_global = self.res_proj_global(self.resnet_global(image)).unsqueeze(1)
        resnet_patch = self.res_proj_patch(self.resnet_patch(patch_batch)).view(B, N, -1)

        combined_features = torch.cat([swin_global, swin_patch, resnet_global, resnet_patch], dim=1)
        pooled = combined_features.mean(dim=1)
        return self.classifier(pooled)


def freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False


def apply_freeze(model: nn.Module, cfg):
    if getattr(cfg.MODEL.FREEZE, 'BACKBONE', False):
        freeze_module(model.swin_global)
        freeze_module(model.swin_patch)
        freeze_module(model.resnet_global)
        freeze_module(model.resnet_patch)
        if hasattr(model, 'patch_resize'):
            freeze_module(model.patch_resize)

    if getattr(cfg.MODEL.FREEZE, 'PROJECTION', False):
        if hasattr(model, 'swin_global_fc'):
            freeze_module(model.swin_global_fc)
        if hasattr(model, 'swin_patch_fc'):
            freeze_module(model.swin_patch_fc)
        if hasattr(model, 'feature_attention'):
            freeze_module(model.feature_attention)
        if hasattr(model, 'res_proj_global'):
            freeze_module(model.res_proj_global)
        if hasattr(model, 'res_proj_patch'):
            freeze_module(model.res_proj_patch)

    if getattr(cfg.MODEL.FREEZE, 'CLASSIFIER', False):
        freeze_module(model.classifier)


def log_freeze_status(model: nn.Module, logger: logging.Logger, name: str = ""):
    logger.info(f"🔍 [Freeze Status] {name}")
    for param_name, param in model.named_parameters():
        status = "🔒 FROZEN" if not param.requires_grad else "✅ TRAINABLE"
        logger.info(f"{status:12} | {name}.{param_name}")



def get_feature_extractor(cfg, is_train, **kwargs):
    if cfg.MODEL.EXTRA.WITH_ATTN:
        model = TwoBranchModelWithAttn(pretrained=True)
    elif cfg.MODEL.EXTRA.ONLYCAT:
        model = TwoBranchModelOnlyCat(pretrained=True)
    elif cfg.MODEL.EXTRA.VIEWCAT:
        model = TwoBranchModelViewCat(pretrained=True)

    apply_freeze(model, cfg)
    
    log_freeze_status(model.swin_global, logger, "Swin Global")
    log_freeze_status(model.resnet_patch, logger, "ResNet Patch")
    log_freeze_status(model.classifier, logger, "Classifier")
    if hasattr(model, "feature_attention"):
        log_freeze_status(model.feature_attention, logger, "Feature Attention")
    if hasattr(model, "res_proj_global"):
        log_freeze_status(model.res_proj_global, logger, "ResNet Proj Global")
    
    model.train() if is_train else model.eval()
    return model

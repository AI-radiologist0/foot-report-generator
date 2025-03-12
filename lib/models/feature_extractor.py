import logging
import torch
import torch.nn as nn
import timm
from torchvision import models
import torch.nn.functional as F


def create_swin_model(variant, pretrained):
    model = timm.create_model(variant, pretrained=pretrained)
    model.head = nn.Identity()
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

    # RAW 모델 선택
    if raw_type == 'swin-t':
        global_model, global_dim = create_swin_model('swin_tiny_patch4_window7_224', pretrained)
    elif raw_type == 'resnet':
        global_model, global_dim = create_resnet_model(pretrained, in_channels=102)
    else:
        raise ValueError(f"Unsupported RAW model type: {raw_type}")

    # PATCH 모델 선택
    if patch_type == 'swin-t':
        local_model, local_dim = create_swin_model('swin_tiny_patch4_window7_224', pretrained)
    elif patch_type == 'resnet':
        local_model, local_dim = create_resnet_model(pretrained, in_channels=102)
    else:
        raise ValueError(f"Unsupported PATCH model type: {patch_type}")

    return global_dim, global_model, local_dim, local_model

class FeatureExtractor(nn.Module):
    def __init__(self, cfg ,pretrained=True, **kwarg):
        super(FeatureExtractor, self).__init__()

        self.target_classes = cfg.DATASET.TARGET_CLASSES
        self.is_binary = len(self.target_classes) == 2
        self.output_dim = 1 if self.is_binary else len(self.target_classes)

        self.global_feature_dim, self.global_branch, self.local_feature_dim, self.local_branch = get_model(cfg, pretrained=pretrained)

        # Validation checks
        assert isinstance(self.global_feature_dim, int), "Global feature dimension must be an integer"
        assert isinstance(self.local_feature_dim, int), "Local feature dimension must be an integer"
        assert isinstance(self.global_branch, nn.Module), "Global branch must be a PyTorch module"
        assert isinstance(self.local_branch, nn.Module), "Local branch must be a PyTorch module"


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


    def forward(self, image, patches):
        # Extract global features (Swin Transformer)
        global_features = self.global_branch.forward_features(image)
        # logging.info(f"{global_features.size()}")
        if global_features.dim() == 4:  # Expected format: (batch, height, width, channel)
            global_features = global_features.mean(dim=[1, 2])  # Perform mean pooling
        elif global_features.dim() == 3:  # Expected format: (batch, num_tokens, channels)
            global_features = global_features.mean(dim=1)  # Pool over tokens -> (32, 7)
        else:
            raise ValueError(f"Unexpected feature dimension: {global_features.shape}")

        # Extract local features (ResNet for patches)
        local_features = self.local_branch(patches)
        local_features = local_features.view(local_features.size(0), -1)  # Flatten

        # logging.info(f"{global_features.size()}, {local_features.size()}")

        # **Ensure both feature tensors have same batch dimension**
        combined_features = torch.cat((global_features, local_features), dim=1)

        if self.is_binary:
            return torch.sigmoid(self.classifier(combined_features))
        
        return self.classifier(combined_features)


class TwoBranchModel(nn.Module):
    def __init__(self, pretrained=True):
        super(TwoBranchModel, self).__init__()

        # Swin Transformer for global and patch processing
        self.swin_global = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained)
        self.swin_global.head = nn.Identity()
        self.swin_patch = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained)
        self.swin_patch.head = nn.Identity()
        self.global_feature_dim = self.swin_global.num_features

        # ResNet-50 for global and patch processing
        resnet_global = models.resnet50(pretrained=pretrained)
        resnet_patch = models.resnet50(pretrained=pretrained)
        self.local_feature_dim = resnet_global.fc.in_features
        
        # Modify input channel for patch processing (102 channels from 34 patches)
        resnet_patch.conv1 = nn.Conv2d(102, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        resnet_global.fc = nn.Identity()
        resnet_patch.fc = nn.Identity()
        self.resnet_global = resnet_global
        self.resnet_patch = resnet_patch

        # Patch Resizing Conv Layer for Swin Transformer
        self.patch_resize = nn.Conv2d(102, 3, kernel_size=1)  # Convert 102 channels → 3 channels

        # Placeholder for classifier, updated dynamically
        self.classifier = nn.Sequential(nn.Linear(1, 1))

    def forward(self, image, patches):
        # Resize patches to (batch_size, 3, 224, 224) using a conv layer + interpolation
        resized_patches = self.patch_resize(patches)  # (batch, 3, 112, 112)
        resized_patches = F.interpolate(resized_patches, size=(224, 224), mode='bilinear', align_corners=False)

        # Swin Transformer features
        swin_global_features = self.swin_global.forward_features(image)
        swin_patch_features = self.swin_patch.forward_features(resized_patches)

        # Pooling for correct dimensions
        if swin_global_features.dim() == 4:
            swin_global_features = swin_global_features.mean(dim=[2, 3])
        elif swin_global_features.dim() == 3:
            swin_global_features = swin_global_features.mean(dim=1)

        if swin_patch_features.dim() == 4:
            swin_patch_features = swin_patch_features.mean(dim=[2, 3])
        elif swin_patch_features.dim() == 3:
            swin_patch_features = swin_patch_features.mean(dim=1)

        # ResNet-50 features
        resnet_global_features = self.resnet_global(image)
        resnet_patch_features = self.resnet_patch(patches)

        # Flatten patch features
        resnet_patch_features = resnet_patch_features.view(resnet_patch_features.size(0), -1)

        # Concatenate all features
        combined_features = torch.cat((swin_global_features, swin_patch_features, 
                                       resnet_global_features, resnet_patch_features), dim=1)
        
        # Update classifier if feature dimensions change
        if combined_features.shape[1] != self.classifier[0].in_features:
            self.classifier = nn.Sequential(
                nn.Linear(combined_features.shape[1], 512),
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
            ).to(image.device)
        
        return self.classifier(combined_features)



def get_feature_extractor(cfg, is_train, **kwargs):
    model = FeatureExtractor(cfg)
    if is_train:
        model.train()
    else:
        model.eval()
    return model
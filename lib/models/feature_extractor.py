import logging
import torch
import torch.nn as nn
import timm
from torchvision import models
import torch.nn.functional as F


def create_swin_model(variant, pretrained):
    model = timm.create_model(variant, pretrained=pretrained)
    # model.head = nn.Identity()
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
        global_model, global_dim = create_resnet_model(pretrained, in_channels=3)
    else:
        raise ValueError(f"Unsupported RAW model type: {raw_type}")

    # PATCH 모델 선택
    if patch_type == 'swin-t':
        local_model, local_dim = create_swin_model('swin_tiny_patch4_window7_224', pretrained)
    elif patch_type == 'resnet':
        local_model, local_dim = create_resnet_model(pretrained, in_channels=3)
    else:
        raise ValueError(f"Unsupported PATCH model type: {patch_type}")

    return global_dim, global_model, local_dim, local_model

class FeatureExtractor(nn.Module):
    def __init__(self, cfg, pretrained=True, **kwarg):
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
        if cfg.MODEL.EXTRA.USE_CKPT and cfg.MODEL.EXTRA.CKPT:
            logging.info(f"feature extractor load ckpt flag: {cfg.MODEL.EXTRA.USE_CKPT}")
            self.load_from_checkpoint(cfg.MODEL.EXTRA.CKPT)
            

    def forward(self, image, patches):
        # Extract global features (Swin Transformer)
        global_features = self.global_branch.forward_features(image)
        if global_features.dim() == 4:  # Expected format: (batch, height, width, channel)
            global_features = global_features.mean(dim=[1, 2])  # Perform mean pooling
        elif global_features.dim() == 3:  # Expected format: (batch, num_tokens, channels)
            global_features = global_features.mean(dim=1)  # Pool over tokens
        else:
            raise ValueError(f"Unexpected feature dimension: {global_features.shape}")
        
        global_features = global_features.unsqueeze(1)  # Add an extra dimension for batch size
        
        
        # Extract local features (ResNet for patches)
        if patches.ndim == 4:
            # Single patch image per sample
            B, C, H, W = patches.size()
            N = 1
            patches = patches.unsqueeze(1)  # (B, 1, C, H, W)로 변환
        else:
            B, N, C, H, W = patches.size()
        patches = patches.view(B * N, C, H, W)  # Reshape patches to (B*N, C, H, W)
        local_features = self.local_branch(patches)
        local_features = local_features.view(B, N, -1)  # Reshape back to (B, N, feature_dim)
        
        # Ensure both feature tensors have the same batch dimension
        combined_features = torch.cat((global_features, local_features), dim=1)
        
        if self.classifier is None:
            return combined_features

        # 기존 분류 목적 분기 유지
        combined_features = torch.cat((global_features.squeeze(1), local_features.mean(dim=1)), dim=1)
        # Apply the classifier (assuming self.classifier is defined elsewhere in your code)
        # combined_features = self.classifier(combined_features)
        if self.is_binary:
            return torch.sigmoid(self.classifier(combined_features))
        
        return self.classifier(combined_features)

    def load_from_checkpoint(self, checkpoint_path, map_location=None):
        """
        Load model weights from a checkpoint.

        Args:
            checkpoint_path (str): Path to the .pth file.
            map_location (str or torch.device, optional): Device to map the checkpoint. Defaults to None.
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.load_state_dict(checkpoint)
        logging.info(f"Model weights loaded from {checkpoint_path}")


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



def get_feature_extractor(cfg, is_train, remove_classifier=False, **kwargs):
    """
    Initialize the feature extractor model.

    Args:
        cfg: Configuration object.
        is_train: Boolean indicating whether the model is in training mode.
        remove_classifier: Boolean indicating whether to remove the classifier layer.
        **kwargs: Additional arguments.

    Returns:
        Feature extractor model.
    """
    model = FeatureExtractor(cfg)
    
    # Optionally remove the classifier
    if remove_classifier and hasattr(model, 'classifier'):
        model.classifier = None  # Remove the classifier layer
    
    if is_train:
        model.train()
    else:
        model.eval()
    
    return model
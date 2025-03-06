import logging
import torch
import torch.nn as nn
from timm import create_model
from torchvision import models


class FeatureExtractor(nn.Module):
    def __init__(self, cfg ,pretrained=True, **kwarg):
        super(FeatureExtractor, self).__init__()

        self.global_branch = create_model('swin_tiny_patch4_window7_224', pretrained=pretrained)
        self.global_branch.head = nn.Identity()
        self.global_feature_dim = self.global_branch.num_features

        resnet = models.resnet50(pretrained=pretrained)
        self.local_feature_dim = resnet.fc.in_features
        resnet.conv1 = nn.Conv2d(102, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.fc = nn.Identity()
        self.local_branch = resnet
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
            
            nn.Linear(64, 1), 
            nn.Sigmoid()
        )


    def forward(self, image, patches):
        # Extract global features (Swin Transformer)
        global_features = self.global_branch.forward_features(image)
        # logging.info(f"{global_features.size()}")
        if global_features.dim() == 4:  # Expected format: (batch, height, width, channel)
            global_features = global_features.mean(dim=[1, 2])  # Perform mean pooling
        elif global_features.dim() == 3:  # Expected format: (batch, num_tokens, channels)
            global_features = global_features.mean(dim=1)  # Pool over tokens -> (32, 7)

        # Extract local features (ResNet for patches)
        local_features = self.local_branch(patches)
        local_features = local_features.view(local_features.size(0), -1)  # Flatten

        # logging.info(f"{global_features.size()}, {local_features.size()}")

        # **Ensure both feature tensors have same batch dimension**
        combined_features = torch.cat((global_features, local_features), dim=1)

        return self.classifier(combined_features)
    

def get_feature_extractor(cfg, is_train, **kwargs):
    model = FeatureExtractor(cfg)
    if is_train:
        model.train()
    else:
        model.eval()
    return model
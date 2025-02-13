# --------------------------------------------------------
# Reference from HRNet-Human-Pose-Estimation
# refact code from old one.
# Written by JeongMin Kim(jm.kim@dankook.ac.kr)
# ----------------------------------------------------

import torch
import torch.nn as nn
from torchvision.ops import MLP


class FCN(nn.Module):
    def __init__(self, cfg):
        super(FCN, self).__init__()
        self.params = cfg.MODEL.EXTRA
        self.mlp = _add_mlp_block(self.params)
        self.output = _add_linear(self.params)

    def forward(self, x):
        x = self.mlp(x)
        x = self.output(x)
        x = nn.functional.log_softmax(x)
        return x


def _add_mlp_block(cfg):
    params = {
        "in_channels": cfg.INPUT_CHANNELS,
        "hidden_channels": cfg.HIDDEN_CHANNELS,
        "dropout": cfg.HIDDEN_DROPOUT,
        "activation_layer": nn.ReLU if cfg.HIDDEN_ACTIVATION == "ReLU" else nn.Tanh
    }
    block = MLP(**params)
    return block


def _add_linear(cfg):
    params = {
        'in_features': cfg.HIDDEN_CHANNELS[-1],
        'out_features': cfg.OUTPUT_CHANNELS
    }
    return nn.Linear(**params)

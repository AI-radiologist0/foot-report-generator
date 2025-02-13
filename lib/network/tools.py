""" Tools for implementing neural networks """
from torch import nn
from torch.nn import Conv2d
from torchvision.ops import MLP


def _add_mlp_block(cfg):
    params = {
        "in_channels": cfg.IN.CHANNELS,
        "hidden_channels": cfg.HIDDEN.CHANNELS,
        "dropout": cfg.HIDDEN.DROPOUT,
        "activation_layer": nn.ReLU if cfg.HIDDEN.ACTIVATION == "ReLU" else nn.Tanh
    }
    block = MLP(**params)
    return block


def _add_linear(block_info):
    params = {
        'in_features': int(block_info["in_channels"]),
        'out_features': int(block_info['out_channels'])
    }
    return nn.Linear(**params)


def _add_conv_block(block_info):
    params = {
        "in_channels": int(block_info["in_channels"]),
        "out_channels": int(block_info["out_channels"]),
        "kernel_size": int(block_info["kernel_size"]),
        "stride": int(block_info["stride"]),
        "padding": int(block_info.get("padding", 0)),  # Default padding is 0 if not specified
    }

    # if block_info["type"] == "ConvBlock":
    #     return ConvBlock(**params)

    return Conv2d(**params)


def _add_pooling_layer(block_info):
    if block_info["method"] == "AdaptAvg":
        return nn.AdaptiveAvgPool2d((1, 1))
    pooling_layer = nn.AvgPool2d if block_info["method"] == "average" else nn.MaxPool2d
    params = {
        "kernel_size": int(block_info["kernel_size"]),
        "stride": int(block_info["stride"]),
        "padding": int(block_info["padding"])
    }

    return pooling_layer(**params)


def _add_dropout(block_info):
    params = {
        "p": float(block_info["dropout_ratio"])
    }
    return nn.Dropout(**params)


def set_layer(config):
    """ set layer from config file """
    module_list = nn.ModuleList()

    # input shape

    
    # iter config files
    # for idx, info in enumerate(config):
    #     if info['type'] == 'MLP':
    #         module_list.append(_add_mlp_block(info))
    #         continue
    #     if info['type'] == 'Output':
    #         module_list.append(nn.LogSoftmax(dim=1))
    #     if info['type'] == 'Conv':
    #         module_list.append(_add_conv_block(info))
    #     if info['type'] == 'Pooling':
    #         module_list.append(_add_pooling_layer(info))
    #     if info['type'] == 'Flatten':
    #         module_list.append(nn.Flatten())
    #     if info['type'] == 'Linear':
    #         module_list.append(_add_linear(info))
    #
    #     if "activation" in info.keys():
    #         if info['activation'] == "relu":
    #             module_list.append(nn.ReLU(inplace=True))
    #         if info['activation'] == "tanh":
    #             module_list.append(nn.Tanh(inplace=True))

    return module_list

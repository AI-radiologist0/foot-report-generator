from torch import nn
from network.tools import set_layer

""" Neural Network template (Using config file) """


class Net(nn.Module):
    def __init__(self, config, num_classes=10):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.layers = set_layer(self.config)

    def forward(self, x):
        print(self.layers)
        x = x.view(x.size(0), -1)
        for idx, layer in enumerate(self.layers):
            x = layer(x)
        return x

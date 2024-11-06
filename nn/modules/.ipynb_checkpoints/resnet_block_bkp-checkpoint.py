import torch
from torch import nn
from base import mlp, conv

__all__=[
    "ResNetBlock"
]


class ResNetBlock(nn.Module):
    def __init__(self, channels, activation, norm , pool, downsample):
        super(ResNetBlock, self).__init__()
        self.downsample = None
        self.conv_block = conv(channels, activation=activation, norm=norm, pool=pool, res_block=True, build_seq=True)
        if downsample:
            self.downsample = nn.Sequential(nn.Conv2d(channels[0], channels[1], kernel_size=1, stride=2), nn.BatchNorm2d(channels[1]))
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x
        out = self.conv_block(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)
        
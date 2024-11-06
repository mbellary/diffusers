import torch
import importlib
from torch import nn

from nn.modules.base import mlp, conv
from nn.modules.convdown_block import ConvDownBlock
from nn.modules.convup_block import ConvUpBlock

__all__ = [
    "UNet"
]

class UNet(nn.Module):
    def __init__(self, 
                 in_channel=1):
        super().__init__()

        self.downsample_blocks = nn.ModuleList([
                                    ConvDownBlock(in_channel=1, out_channels=[64, 64], activation='ReLU', norm='BatchNorm2d',   pool='MaxPool2d'),
                                    ConvDownBlock(in_channel=64, out_channels=[128, 128], activation='ReLU', norm='BatchNorm2d', pool='MaxPool2d'),
                                    ConvDownBlock(in_channel=128, out_channels=[256, 256], activation='ReLU', norm='BatchNorm2d', pool='MaxPool2d'),
                                    ConvDownBlock(in_channel=256, out_channels=[512, 512], activation='ReLU', norm='BatchNorm2d', pool='MaxPool2d')])
        self.bottleneck_block = conv(features=[512, 1024, 1024], kernel_sizes=[3, 3], strides=[1, 1], paddings=[0, 0], build_seq=True)
        self.upsample_blocks = nn.ModuleList([
                                    ConvUpBlock(in_channel=1024, out_channels=[512, 512, 512]),
                                    ConvUpBlock(in_channel=512, out_channels=[256, 256, 256]),
                                    ConvUpBlock(in_channel=256, out_channels=[128, 128, 128]),
                                    ConvUpBlock(in_channel=128, out_channels=[64, 64, 64]) ])
        self.output = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)
        
    def forward(self, x):
        block_state = [] # output of the last conv in a block.
        for ds_block in self.downsample_blocks:
            x, state = ds_block(x)
            block_state.append(state)
        block_state = list(reversed(block_state))
        x = self.bottleneck_block(x)
        for i, us_block in enumerate(self.upsample_blocks):
            x = us_block(x, block_state[i])
        out = self.output(x)
        return out
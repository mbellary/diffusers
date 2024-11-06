import torch
import importlib
from torch import nn

from nn.modules.resnet_block import ResNetBlock

__all__ = [
    "PixelCNN"
]

class PixelCNN(nn.Module):
    def __init__(self, 
                 channels, 
                 kernel_sizes,
                 strides,
                 paddings,
                 convs_per_block, 
                 num_blocks, 
                 activation, 
                 in_channels):
        super(PixelCNN, self).__init__()
        self.channels = []
        self.layers = 48
        self.channels = ([channels * 2] + [channels] * 2 + [channels * 2]) * (num_blocks -1)
        # Mask A Causal
        self.network = [CausalConv2D('A', in_channels=in_channels, out_channels=channels*2, kernel_size=7, stride=1, padding=3),
                        nn.ReLU()]
        
        # # Mask B Causal
        for i in range(0, len(self.channels), convs_per_block+1): # generates res blocks
            in_channels = self.channels[i : i+convs_per_block+1]            
            self.network += [ResNetBlock(self.layers, 
                                         in_channels, 
                                         kernel_sizes,
                                         strides,
                                         paddings, 
                                         activation,
                                         causal='B')]
        self.network += [nn.Conv2d(channels * 2, channels * 2, kernel_size=1), nn.Flatten(start_dim=2)]
        self.network = nn.Sequential(* self.network)

    def forward(self, x):
        x = self.network(x)
        return torch.permute(x, (0, 2, 1))
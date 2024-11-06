import torch
from torch import nn
from torch import tensor

from nn.modules.base import mlp, conv
from nn.modules.resnet_block import ResNetBlock
from attentions.multihead_attention import MultiheadAttention

__all__ = [
    "ConvDownBlock"
]

class ConvDownBlock(nn.Module):
    def __init__(self, 
                 in_channel,
                 out_channels,
                 num_blocks = 2,
                 convs_per_block = 2,
                 activation='SiLU',
                 norm='GroupNorm',
                 pool=None,
                 time_embed_channel=512, 
                 num_groups=32,
                 num_heads = 4,
                 block_type='conv',
                bottleneck = False):
        super().__init__()
        self.block_type = block_type
        self.has_attention = False
        self.is_bottleneck = bottleneck

        if self.block_type == 'attention':
            d_model = out_channels[0] # d_model must be the out_channel dim.
            d_q = d_k = d_v = d_model // num_heads
            self.self_attention = [MultiheadAttention(d_model, 
                                                      d_q, 
                                                      d_k, 
                                                      d_v, 
                                                      num_heads, 
                                                      'conv') for i in range(num_blocks)]
            self.block_type = 'resnet' # attention block requires resnet and downsample blocks.
            self.has_attention = True
        
        if self.block_type == 'resnet':
            channels = [in_channel] + out_channels * num_blocks
            mid = len(channels) // 2
            block_channels =  [channels[:mid+1]] + [channels[mid:]]
            self.layers = [ResNetBlock(in_channel= block_channels[i][0],
                                     out_channels = block_channels[i][1:],
                                     kernel_sizes = [3, 3],
                                     convs_per_block = convs_per_block,
                                     activation= activation, 
                                     norm = norm,
                                     num_groups = num_groups,
                                     time_embed_channels = time_embed_channel,
                                     block_type = self.block_type) for i in range(num_blocks)] # downsampling halves the resolution map
            self.downsample = nn.Conv2d(channels[1], channels[-1], kernel_size=3, stride=2, padding=1) if not self.is_bottleneck else None
            
        if self.block_type == 'conv':
            self.layers = conv(features= [in_channel] + out_channels, 
                               kernel_sizes=[3, 3], 
                               strides=[1, 1], 
                               paddings=[0, 0], 
                               activation=activation, 
                               norm=norm)
            self.pool = nn.MaxPool2d(kernel_size=2)
        self.layers = nn.ModuleList(self.layers)
            
    def forward(self, x, timesteps=None):
        downsampled = x
        for i, layer in enumerate(self.layers):
            if self.block_type == 'conv': x = layer(x) # conv block
            elif self.block_type == 'resnet' and self.has_attention: # attention block
                x = layer(x, timesteps)
                x = self.self_attention[i](x)
            else: x = layer(x, timesteps) # resnet block
        if self.block_type == 'resnet' and not self.is_bottleneck:
            downsampled = self.downsample(x) # downsampling w/ stride
        if self.block_type == 'conv':
            downsampled = self.pool(x) # downsampling w/ pooling
        return downsampled, x
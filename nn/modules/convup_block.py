import torch
from torch import nn
from torch import tensor

from nn.modules.base import mlp, conv
from nn.modules.resnet_block import ResNetBlock
from attentions.multihead_attention import MultiheadAttention
from torchvision.transforms.functional import center_crop


__all__ = [
    "ConvUpBlock"
]

class ConvUpBlock(nn.Module):
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
            channels = [out_channels[0] * 2] + out_channels * num_blocks
            mid = len(channels) // 2
            block_channels =  [channels[:mid+1]] + [channels[mid:]]
            # Upsample first with conv
            #self.skip_connection = True if in_channel != out_channels[0] else False
            self.upsample_layer = conv(upsample = True, 
                                       features= [in_channel] + [out_channels[0]], 
                                       kernel_sizes=[3],
                                       strides=[2], 
                                       paddings=[1], 
                                       output_paddings=[1], 
                                       activation=activation,
                                       norm=norm,
                                       num_groups = num_groups,
                                       pool=pool,
                                       build_seq=True)

            # downsample with conv
            self.conv_layers = [ResNetBlock(upsample = False,
                                    in_channel= block_channels[i][0],
                                     out_channels = block_channels[i][1:],
                                     kernel_sizes = [3] * convs_per_block,
                                     convs_per_block = convs_per_block,
                                     activation= activation, 
                                     norm = norm,
                                     num_groups = num_groups,
                                     time_embed_channels = time_embed_channel,
                                      block_type = self.block_type) for i in range(num_blocks)]
            
        if self.block_type == 'conv':
            self.upsample_layer = conv(upsample = True, 
                                       features= [in_channel] + [out_channels[0]], 
                                       kernel_sizes=[3],
                                       strides=[2], 
                                       paddings=[1], 
                                       output_paddings=[1], 
                                       activation=activation,
                                       norm=norm,
                                       num_groups = num_groups,
                                       pool=pool,
                                       build_seq=True)
            self.conv_layers =  conv(upsample = False, 
                                       features= [in_channel] + out_channels[1:], 
                                       kernel_sizes=[3, 3],
                                       strides=[1, 1],
                                       paddings=[0, 0] , 
                                       output_paddings=[0, 0], 
                                       activation=activation,
                                       norm=norm,
                                       num_groups = num_groups,
                                       pool=pool,
                                       build_seq=True)
            
    def forward(self, x, downsample_state):
        x = self.upsample_layer(x)
        downsample_cropped = center_crop(downsample_state, x.shape[2])
        x = torch.cat((x, downsample_cropped), dim=1) #if self.skip_connection else x
        for layer in self.conv_layers:
            x = layer(x)
        return x
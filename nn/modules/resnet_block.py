import torch
from torch import nn
from nn.modules.base import mlp, conv

__all__=[
    "ResNetBlock"
]

class ResNetBlock(nn.Module):
    def __init__(self, 
                 in_channel,
                 out_channels,
                 kernel_sizes, 
                 convs_per_block=2,
                 activation='ReLU', 
                 norm='BatchNorm2d', 
                 pool=None, 
                 causal=None,
                 num_groups = 32,
                 time_embed_channels = 512,
                 layers=None,
                upsample=False,
                block_type='conv'):
        super(ResNetBlock, self).__init__()

        self.block_type = block_type
        self.channels = [in_channel] + out_channels
        self.downsample = True if (in_channel != out_channels[0] and self.block_type=='conv') else False
        strides = [2] + ([1] * convs_per_block) if self.downsample else [1] * convs_per_block  #in:[64] out[128, 128]
        paddings =  [1, 1]
        self.shortcut_connection = nn.Conv2d(in_channel, out_channels[-1], kernel_size=1, stride=2)
        self.conv_blocks = nn.ModuleList(conv(self.channels, 
                               kernel_sizes=kernel_sizes,
                               strides = strides,
                               paddings=paddings,
                               activation=activation, 
                               norm=norm, 
                               pool=pool, 
                               res_block=True, 
                               res_layer=layers,
                               causal=causal,
                               num_groups=num_groups,
                               upsample=upsample))
        
        self.position_time_embedding = nn.Linear(time_embed_channels, out_channels[0])
        self.residual_connection = nn.Conv2d(self.channels[1], self.channels[1], kernel_size=1)

    def forward(self, x, timesteps=None):
        identity = x
        for i, layer in enumerate(self.conv_blocks):
            x = layer(x)
            if i == 0 and timesteps is not None:
                time_embed = self.position_time_embedding(timesteps)
                time_embed = time_embed[:, :, None, None]
                x = time_embed + x
        identity = self.shortcut_connection(x) if self.downsample and self.block_type == 'conv' else self.residual_connection(x)
        x += identity
        return x
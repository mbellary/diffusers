import torch
import importlib
from torch import nn

__all__ = [
    "mlp",
    "conv"
]
nn_cls = importlib.import_module("torch.nn")

def mlp(features, activation=None, build_seq=False, *args, **kwargs):
    layers = []        
    layer = [(nn.Linear(features[i], features[i+1])) for i in range(len(features) - 1)]
    for module in layer:
        layers.append(module)
        if activation :
            act_fn = getattr(nn_cls, activation)
            layers.append(act_fn())
    if not build_seq:
        return layers
    return nn.Sequential(*layers)

class CausalConv2D(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        out_filters, in_filters , filter_height, filter_width = self.weight.shape
        mask = torch.ones_like(self.weight)
        height = filter_height//2
        width = filter_width//2 + 0 if mask_type == 'A' else height + 1
        mask.data[:, :, height+1:, :] = 0
        mask.data[:, :, height:, width:] = 0
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

# ks, strides, paddings, op must be list , provided by user.
def conv(features, kernel_sizes=3, 
                     strides=1, 
                     paddings=0, 
                     output_paddings=0, 
                     activation=None, 
                     norm=None, 
                     pool=None, 
                     upsample=False, 
                     build_seq=False, 
                     res_block=False, 
                     res_layer=32,
                     causal=None,
                     num_groups = None,
                     *args, 
                     **kwargs):
    layers = []
    len_feats = len(features) - 1
    conv_blocks = [nn.Conv2d(features[i], 
                             features[i+1], 
                             kernel_size=(kernel_sizes[i], kernel_sizes[i]), 
                             stride=(strides[i], strides[i]), 
                             padding=(paddings[i], paddings[i])) for i in range(len_feats)]
    
    if res_block:
        blocks = []
        for i in range(len_feats):
            blocks += [nn.Conv2d(features[i], 
                                 features[i+1], 
                                 kernel_size=kernel_sizes[i], 
                                 stride=(strides[i], strides[i]), 
                                 padding=(paddings[i], paddings[i]))]
            if i == 1 and causal:
                blocks.pop()
                blocks += [CausalConv2D(causal,
                                 in_channels=features[i], 
                                 out_channels=features[i+1], 
                                 kernel_size=kernel_sizes[i], 
                                 stride=(strides[i], strides[i]), 
                                 padding=(paddings[i], paddings[i]))]
        conv_blocks = blocks
        
    if upsample:
        conv_blocks = [nn.ConvTranspose2d(features[i], 
                                          features[i+1], 
                                          kernel_size=kernel_sizes[i], 
                                          stride=strides[i], 
                                          padding=paddings[i], 
                                          output_padding=output_paddings[i]) for i in range(len_feats)]
    
    for i, module in enumerate(conv_blocks):
        layers.append(module)
        if res_block :
            if i == len(conv_blocks)-1 and res_layer == 32: activation = None
            activation = None if i != len(conv_blocks)-1 and res_layer == 50 else activation
        if norm:
            norm_fn = getattr(nn_cls, norm)
            if num_groups:
                layers.append(norm_fn(num_groups, module.out_channels))
            else: layers.append(norm_fn(module.out_channels))
        if activation:
            act_fn = getattr(nn_cls, activation)
            layers.append(act_fn())
        if pool and i == len(conv_blocks)-1: # applying pooling only after the last conv block. #reevaluate
            pool_fn = getattr(nn_cls, pool)
            layers.append(pool_fn(2))  # kernel_size is fixed            
    if not build_seq:
        return layers
    
    return nn.Sequential(*layers)
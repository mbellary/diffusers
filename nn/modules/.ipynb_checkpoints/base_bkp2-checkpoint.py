import torch
from torch import nn

__all__ = [
    "mlp",
    "conv"
]
nn_cls = importlib.import_module("torch.nn")

def mlp(features, activation=None, build_seq=False):
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

def conv(features, kernel_size=3, stride=1, padding=0, activation=None, norm=None, pool=None, decoder=False, build_seq=False, res_block=False):
    layers = []
    len_feats = len(features) - 1
    
    conv_blocks = [nn.Conv2d(features[i], features[i+1], kernel_size=kernel_size, stride=stride, padding=padding) for i in range(len_feats)]

    if res_block:
        blocks = []
        for i in range(len_feats):
            if features[i] != features[i+1] : stride = 2
            else: stride = 1
            blocks += [nn.Conv2d(features[i], features[i+1], kernel_size=3, stride=stride, padding=1)]
        conv_blocks = blocks
        
    if decoder:
        conv_blocks = [nn.ConvTranspose2d(features[i], features[i+1], kernel_size=kernel_size, stride=stride, padding=padding, output_padding=1) for i in range(len_feats)]
    
    for i, module in enumerate(conv_blocks):
        layers.append(module)
        if res_block :
            if i == len(conv_blocks)-1: activation = None
        if norm:
            norm_fn = getattr(nn_cls, norm)
            layers.append(norm_fn(module.out_channels))
        if activation:
            act_fn = getattr(nn_cls, activation)
            layers.append(act_fn())
    
    if not build_seq:
        return layers
    
    return nn.Sequential(*layers)
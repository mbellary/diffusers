import importlib
import torch
from torch import nn

__all__ = [
    "mlp",
    "conv"
]
nn_cls = importlib.import_module("torch.nn")

def mlp(features, activation=None):
    layers = []        
    layer = [(nn.Linear(features[i], features[i+1])) for i in range(len(features) - 1)]
    for module in layer:
        layers.append(module)
        if activation :
            act_fn = getattr(nn_cls, activation)
            layers.append(act_fn()) 
    return nn.Sequential(*layers)

def conv(features, kernel_size=3, stride=1, padding=0, activation=None, norm=None, decoder=False):
    layers = []
    conv_block = [nn.Conv2d(features[i], features[i+1], kernel_size=kernel_size, stride=stride, padding=padding) for i in range(len(features)-1)]
    if decoder:
        conv_block = [nn.ConvTranspose2d(features[i], features[i+1], kernel_size=kernel_size, stride=stride, padding=padding, output_padding=1) for i in range(len(features)-1)]
    #layer = [conv_block(features[i], features[i+1], kernel_size=kernel_size, stride=stride, padding=padding) for i in range(len(features)-1)]
    layer = conv_block
    for module in layer:
        layers.append(module)
        if norm:
            norm_fn = getattr(nn_cls, norm)
            layers.append(norm_fn(module.out_channels))
        if activation:
            act_fn = getattr(nn_cls, activation)
            layers.append(act_fn())
    return nn.Sequential(*layers)
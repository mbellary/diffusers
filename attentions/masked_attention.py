import torch
from torch import nn

__all__ = [
    "CausalConv2D"
]

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
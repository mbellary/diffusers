import torch
from torch import nn

__all__ = [
    "GatedActivation"
]

class GatedActivation(nn.Module):
    def __init__(self, activation_fn = nn.Identity()):
        super().__init__()
        self._activation_fn = activation_fn

    def forward(self, x):
        _, c, _, _ = x.shape
        x, gated = x[:, c//2:, :, :], x[:, :c//2, :, :]
        return self._activation_fn(x) * torch.sigmoid(gated)
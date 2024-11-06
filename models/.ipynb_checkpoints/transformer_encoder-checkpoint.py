import torch
import importlib
from torch import nn

from models.encoder import EncoderBlock
from encoders.position_encoder import PositionEncoder


class TransformerEncoder(nn.Module):
    def __init__(self, n_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([EncoderBlock(512, 512, 8) for _ in range(n_blocks)])
        self.pe = PositionEncoder(512)

    def forward(self, x, poistional_encoding=False):
        if positional_encoding:
            x = self.pe(x)
        for block in self.blocks:
            x = block(x) # output of block N is passed as input to block N+1
        return x
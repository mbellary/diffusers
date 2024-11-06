import torch
import importlib
from torch import nn

from attentions.multihead_attention import MultiheadAttention


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.mha = MultiheadAttention(input_dim, embed_dim, num_heads)
        self.FFN = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                    nn.Linear(embed_dim, embed_dim))
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_values = self.mha(x)
        x = x + self.dropout(attn_values)
        norm1 = self.layer_norm1(x)
        
        ffn = self.FFN(norm1)
        norm1 = norm1 + self.dropout(ffn)
        out = self.layer_norm2(norm1)
        return out
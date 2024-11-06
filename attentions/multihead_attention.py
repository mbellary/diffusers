import torch
import math
from torch import nn
from torch.nn import functional as F
from torch import tensor

from nn.modules.base import mlp, conv


__all__ = [
    "scaled_dot_attention",
    "QKVOBuilder",
    "MultiheadAttention"
]

def scaled_dot_attention(q, k, v, mask=None, is_causal=False, dropout=None):
    T, S = q.size(-2), k.size(-2)
    d_k  = q.size(-1)
    causal_mask = torch.ones(T, S)
    ## ToDo
        # attention mask
        # dropout
    if is_causal:
        assert mask is None, "attention mask must be None"
        temp_mask = torch.tril(torch.ones(T, S), diagonal=-1)
        causal_mask.masked_fill(temp_mask == 0, float('-inf'))
    atten_logits = q.matmul(k.transpose(-2, -1)) / math.sqrt(d_k)
    atten_logits += causal_mask
    atten_scores = F.softmax(atten_logits, dim=-1)
    values = atten_scores.matmul(v)
    return values


class QKVOBuilder(nn.Module):
    def __init__(self, input_dim, num_heads, q_dim, k_dim, v_dim, kernel_size=1, padding=0, module_type='conv'):
        super().__init__()
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        layer = conv
        if module_type == 'mlp':
            layer = mlp
        self.q = layer([input_dim, q_dim], activation=None, kernel_sizes=[kernel_size], strides=[1], paddings=[padding])[0]
        self.k = layer([input_dim, k_dim], activation=None, kernel_sizes=[kernel_size], strides=[1], paddings=[padding])[0]
        self.v = layer([input_dim, v_dim], activation=None, kernel_sizes=[kernel_size], strides=[1], paddings=[padding])[0]
        # output dim must be the embedding dimension
        self.cv = layer([v_dim, input_dim],  activation=None, kernel_sizes=[kernel_size], strides=[1], paddings=[padding])[0]

    def forward(self, x, dims, batch_size, seq_len, num_heads=1):
        q = self.q(x).reshape(batch_size, seq_len, num_heads, self.q_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(batch_size, seq_len, num_heads, self.k_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(batch_size, seq_len, num_heads, self.v_dim).permute(0, 2, 1, 3)
        return q, k, v, self.cv

   
class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, q_dim, k_dim, v_dim, num_heads, module_type='conv'):
        super().__init__()
        self.input_dim = input_dim
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.module_type = module_type
        #self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.qkvo_builder = QKVOBuilder(input_dim, 
                                      num_heads=num_heads, 
                                      q_dim=self.q_dim,
                                      k_dim=self.k_dim,
                                      v_dim=self.v_dim,
                                      kernel_size=1,
                                      module_type=self.module_type)
    # To Do
    # Weights initializations
    def forward(self, x, mask=None, is_causal=False, dropout=False):
        dims = len(x.shape)
        batch_size, seq_len = (x.shape[0], x.shape[-1] * x.shape[-2]) if dims == 4 else (x.shape[0], x.shape[1])
        query, key, value, context_vector = self.qkvo_builder(x, dims, batch_size, seq_len)
        
        #ToDo :  masking must be done only during training.
        values = scaled_dot_attention(query, key, value, mask=mask, is_causal=is_causal)
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(batch_size, self.v_dim, int(math.sqrt(seq_len)), int(math.sqrt(seq_len))) if dims == 4 else values.reshape(batch_size, seq_len, self.v_dim)
        output = context_vector(values)
        return output
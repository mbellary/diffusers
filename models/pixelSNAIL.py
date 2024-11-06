import torch
from torch import nn

from attentions.multihead_attention import  MultiheadAttention
from attentions.masked_attention import CausalConv2D
from activations.gated_activation import GatedActivation



def conv1x1(x):
    return nn.Conv2d(256, 256, kernel_size=1)(x)

def _elu_conv_elu(x):
    x = F.elu(conv1x1(F.elu(x)))
    return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self._elu_conv_elu = _elu_conv_elu
        self.conv = nn.Conv2d(in_channel, out_channel * 2, kernel_size=3, padding=1)
        self.glu = GatedActivation()

    def forward(self, x):
        elu2 = self._elu_conv_elu(x)
        glu_out = self.glu(self.conv(elu2))
        return x + glu_out

class PixelSNAILBlock(nn.Module):
    def __init__(self, 
                 in_channel=256,
                 q_dim = 32,
                 k_dim = 32,
                 v_dim = 256,
                 num_heads = 1,
                 res_blocks = 4):
        super().__init__()
        self.residual_blocks = nn.Sequential(*[ResnetBlock(in_channel, in_channel) for _ in range(res_blocks)])
        self.attention_block = MultiheadAttention(in_channel, q_dim, k_dim, v_dim, num_heads, 'conv')
        self._elu_conv_elu = _elu_conv_elu

    def forward(self, x, mask=None, is_causal=True):
        res_block = self.residual_blocks(x)
        att_block = self.attention_block(res_block, mask, is_causal)
        out = self._elu_conv_elu(self._elu_conv_elu(res_block) + self._elu_conv_elu(att_block))
        return out

class PixelSNAIL(nn.Module):
    def __init__(self, in_channel, q_dim, k_dim, v_dim, num_heads):
        super().__init__()
        self.input = CausalConv2D('A', in_channels=in_channel, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pixelsnail_blocks = nn.ModuleList([PixelSNAILBlock(256, q_dim = 32,k_dim = 32,v_dim = 256,num_heads = 1) for _ in range(6)])
        self.output = nn.Sequential(nn.Conv2d(256, 10*10, kernel_size=1), nn.ELU())

    def forward(self, x, mask=None, is_causal=True):
        inp = self.input(x)
        for psb in self.pixelsnail_blocks:
            inp = psb(inp, mask, is_causal)
        output = self.output(inp)
        return output
import torch
from torch import nn

__all__ = [
    "PositionEncoder"
]


class PositionEncoder(nn.Module):
    def __init__(self, embed_dim, max_timesteps=1000):
        super().__init__()
        assert embed_dim % 2 == 0 , "Embedding dimension must be even number"
        self.pe = torch.zeros(max_timesteps, embed_dim) #timesteps x dims
        even_indices = torch.arange(0, embed_dim, 2)
        timesteps = torch.arange(0, max_timesteps).float().unsqueeze(1)
        log_term = torch.log(torch.tensor(10000)) / embed_dim
        div = torch.exp(even_indices * -log_term)
        self.pe[:, 0::2] = torch.sin(timesteps * div)
        self.pe[:, 1::2] = torch.cos(timesteps * div)

    def forward(self, timesteps):
        # timesteps : timestep_indices x 1
        #x = x + self.pe[:, :x.size(1)] # ToDO: transformer implementation

        # DDPM implementation
        x = self.pe[timesteps]  # timestep_indices x embed_dim
        return x 


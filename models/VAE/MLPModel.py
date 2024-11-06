import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from nn.modules.base import mlp, conv
from torch.distributions.multivariate_normal import MultivariateNormal

__all__ = [
    "VAEModel"
]
class VAEModel(nn.Module):
    def __init__(self, 
                 in_feature, 
                 encoder_units, 
                 decoder_units, 
                 z_dim, 
                 img_height, 
                 img_width,
                activation):
        super(VAEModel, self).__init__()
        self.encoder = mlp([in_feature] + encoder_units , activation, build_seq=True).append(nn.Linear(encoder_units[-1], (2 * z_dim)))
        self.softplus = nn.Softplus()
        self.decoder = mlp([z_dim] + decoder_units, activation, build_seq=True).append(nn.Linear(decoder_units[-1], (img_height * img_width))).append(nn.Sigmoid())
        
    def forward(self, x):
        latent_dist = self.encode(x)
        z = self.reparamv2(latent_dist)
        recons_x = self.decode(z)
        return {'recons_x' : recons_x, 'latent_dist' : latent_dist, 'z' : z}
        
    def encode(self, x, eps=1e-8):
        out = self.encoder(x)
        mu, log_var = torch.chunk(out, 2, dim=-1)
        scaled = self.softplus(log_var) + eps
        latent_dist = MultivariateNormal(mu, scale_tril=torch.diag_embed(scaled))
        return latent_dist

    def decode(self, z):
        recons_x = self.decoder(z)
        return recons_x
    
    def reparametrization(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        return z

    def reparamv2(self, latent_dist):
        return latent_dist.rsample()
import torch
from torch import nn
from nn.modules.base import conv
from torch.distributions.multivariate_normal import MultivariateNormal

__all__ = [
    'VAEConvModel'
]

class VAEConvModel(nn.Module):
        def __init__(self, 
                 in_channel, 
                 encoder_units, 
                 decoder_units, 
                 z_dim,
                 kernel_sizes,
                 strides,
                 paddings,
                output_paddings,
                activation,
                norm):
            super(VAEConvModel, self).__init__()
            self.encoder = conv([in_channel] + encoder_units , kernel_sizes = kernel_sizes, strides=strides, paddings=paddings, activation=activation, norm = norm, build_seq=True)
            self.latent_params = nn.Linear(encoder_units[-1] * 2 * 2, (2 * z_dim))
            self.softplus = nn.Softplus()
            self.decoder_input = nn.Linear(z_dim, decoder_units[0] * 2 * 2)
            self.decoder_layers  = nn.Sequential(conv(decoder_units, kernel_sizes=kernel_sizes, strides=strides, paddings=paddings, output_paddings=output_paddings, activation=activation, norm = norm, upsample=True,  build_seq=True),
                                nn.ConvTranspose2d(decoder_units[-1], 1, stride=2, padding=1, output_padding=1, kernel_size=3),
                                nn.Conv2d(1, 1, kernel_size=5),
                                nn.Sigmoid())
        
        def forward(self, x):
            latent_dist = self.encode(x)
            z = self.reparametrize(latent_dist)
            recons_x = self.decode(z)
            return {'recons_x' : recons_x, 'latent_dist' : latent_dist, 'z' : z}
            
        def encode(self, x, eps = 1e-8):
            x = self.encoder(x)
            x = torch.flatten(x, start_dim=1)
            latent_params = self.latent_params(x)
            mu, log_var = torch.chunk(latent_params, 2, dim=-1)
            smoothed = self.softplus(log_var) + eps
            return MultivariateNormal(mu, torch.diag_embed(smoothed))

        def decode(self, z):
            z = self.decoder_input(z)
            z = z.view(-1, 256, 2, 2)
            recons_x = self.decoder_layers(z)
            return recons_x
            
        def reparametrize(self, dist):
            return dist.rsample()
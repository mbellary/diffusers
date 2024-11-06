import torch
import torch.nn.functional as F
from torcheval.metrics import MulticlassAccuracy
from torch import optim
from torch import nn
from torch.distributions.kl import kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal

__all__ = [
    "VAELoss",
    "DDPMLoss"
]

class VAELoss(nn.Module):
    def __init__(self, batch_size):
        super(VAELoss, self).__init__()
        self.batch_size = batch_size

    def forward(self, x, recons_x=None, latent_dist=None, z=None):
        prior = self.prior_normal(z)
        recons_loss = self._reconstruction_loss(recons_x, x)
        kl_loss = self._kl_loss_v2(latent_dist, prior)
        loss = recons_loss +  kl_loss
        return {'loss' : loss, 'recons_loss' : recons_loss, 'kl_loss': kl_loss}

    def _reconstruction_loss(self, recons_x, x):
        recons_loss = nn.functional.binary_cross_entropy(recons_x, x, reduction="none").sum(-1).mean()
        return recons_loss
    
    def _kl_loss(self, mu, log_var):
        kl_loss =  - 0.5 * torch.sum(1 + log_var - (mu ** 2) - log_var.exp())
        return kl_loss

    def _kl_loss_v2(self, latent_dist, prior):
        kl_loss = kl_divergence(latent_dist, prior).mean()
        return kl_loss

    def prior_normal(self, z):
        mu = torch.zeros_like(z)
        var = torch.diag_embed(torch.ones_like(z))
        return MultivariateNormal(mu, scale_tril=var)

        
class DDPMLoss(nn.Module):
    def __init__(self, batch_size):
        super(DDPMLoss, self).__init__()
        self.batch_size = batch_size

    def forward(self, noise, pred_noise):
        mse_loss = self._mse_loss(pred_noise, noise)
        return {'loss' : mse_loss}

    def _mse_loss(self, pred_noise, noise):
        mse_loss = nn.functional.mse_loss(pred_noise, noise)
        return mse_loss
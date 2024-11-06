import torch
import importlib
from torch import nn

from nn.modules.base import mlp, conv
from nn.modules.convdown_block import ConvDownBlock
from nn.modules.convup_block import ConvUpBlock
from encoders.position_encoder import PositionEncoder

__all__ = [
    "DDPM",
]

class DDPM():
    def __init__(self,  beta_min=1e-4, beta_max=0.02, max_timesteps=1000):
        self.max_timesteps = max_timesteps
        self.betas = torch.linspace(beta_min, beta_max, max_timesteps)
        self.alphas = 1 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def model(self, in_channel):
        model = _Net(in_channel)
        return model
    
    def forward(self, images, timesteps):
        gausian_noise = torch.randn(images.shape)
        alpha_hat_batch = self.alpha_hat[timesteps][:, None, None, None]
        noisy_image = torch.sqrt(alpha_hat_batch) * images + torch.sqrt(1 - alpha_hat_batch) * gausian_noise
        return noisy_image, gausian_noise

    def reverse(self, model, noisy_image, timesteps):
        predicted_noise = model(noisy_image, timesteps)
        return predicted_noise

    def sampling(self, noisy_image):
        for timestep in range(self.max_timesteps, -1, -1):
            timesteps = timestep * torch.ones(noisy_image.shape[0])
            pred_noise = model(noisy_image, timesteps)
            beta_t = self.betas[timestep]
            alpha_t = self.alphas[timestep]
            alpha_hat_t = self.alpha_hat[timestep]
            alpha_hat_prev = alpha_hat_t[timestep - 1]
            beta_hat_t = (1 - alpha_hat_prev) / (1 - alpha_hat_t) * beta_t
            variance = beta_hat_t * torch.randn(noisy_image.shape) if timesteps > 1 else 0
            noisy_img_prev = torch.pow(alpha_t, - 0.5) * (noisy_image - ((1 - alpha_t) / torch.sqrt(1 - alpha_t) * pred_noise)) + variance

class _Net(nn.Module):
    def __init__(self, in_channel=3):
        super().__init__()
        self.position_time_encoding = nn.Sequential(
                        PositionEncoder(embed_dim=128, max_timesteps=1000),
                        nn.Linear(128, 512),
                        nn.GELU(),
                        nn.Linear(512, 512))
        self.input_conv = nn.Conv2d(in_channel, 128, kernel_size=3, stride=1, padding=1)
        self.downsample_conv = nn.ModuleList([
                                ConvDownBlock(128, [128, 128], block_type='resnet', num_groups=32, time_embed_channel=512),
                                ConvDownBlock(128, [128, 128], block_type='resnet', num_groups=32, time_embed_channel=512),
                                ConvDownBlock(128, [256, 256], block_type='resnet', num_groups=32, time_embed_channel=512),
                                ConvDownBlock(256, [256, 256], block_type='attention', num_heads=4, num_groups=32, time_embed_channel=512),
                                ConvDownBlock(256, [512, 512], block_type='resnet', num_groups=32, time_embed_channel=512)])
        self.bottelneck_conv = ConvDownBlock(512, [512, 512], block_type='attention', num_heads=4, num_groups=32, time_embed_channel=512, bottleneck=True)
        self.upsample_conv = nn.ModuleList([
                                ConvUpBlock(512, [512, 512], block_type='resnet', num_groups=32, time_embed_channel=512),
                                ConvUpBlock(512, [256, 256], block_type='attention', num_heads=4, num_groups=32, time_embed_channel=512),
                                ConvUpBlock(256, [256, 256], block_type='resnet', num_groups=32, time_embed_channel=512),
                                ConvUpBlock(256, [128, 128], block_type='resnet', num_groups=32, time_embed_channel=512),
                                ConvUpBlock(128, [128, 128], block_type='resnet', num_groups=32, time_embed_channel=512)])
        self.output_conv = nn.Sequential(
                            nn.GroupNorm(32, 128),
                            nn.SiLU(),
                            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1))
        
    def forward(self, x, timesteps):
        block_state = []
        time_encoded = self.position_time_encoding(timesteps)
        out = self.input_conv(x)
        for ds in self.downsample_conv:
            out, state = ds(out, time_encoded)
            block_state.append(out)
        out ,_ = self.bottelneck_conv(out, time_encoded)
        block_state = list(reversed(block_state))
        for us, state in zip(self.upsample_conv, block_state):
            out = us(out, state)
        out = self.output_conv(out)
        return out
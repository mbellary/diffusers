from trainer import Trainer, context
import torch
from collections import Counter

__all__ = [
    "DDPMTrainer"
]

class DDPMTrainer(Trainer):
    
    def __init__(self, ddpm, **kwargs):
        super(DDPMTrainer, self).__init__(**kwargs)
        self.mse_loss = []
        self.ddpm = ddpm
    
    def predict(self):
        timesteps = torch.randint(0, self.ddpm.max_timesteps, (self.batch_size,))
        self.noisy_image, self.noise = self.ddpm.forward(self.data, timesteps)
        self.preds = self.ddpm.reverse(self.model, self.noisy_image, timesteps)
            
    def calc_loss(self):
        self.batch_loss = self.loss_fn(self.noise, self.preds)
        self.loss = self.batch_loss['loss']

    def update_grad(self):
        self.loss.backward()
        self.optim.step()

    def collate_preds(self):
        self.mse_loss.append(self.preds['loss'])
        # pass-through to sampling
        self.ddpm.sampling(self.noisy_image)

    @context('eval')
    def eval(self, epochs=1):
        self.model.infer = True
        for epoch in self.epochs:
            with torch.no_grad():self.one_epoch(False)

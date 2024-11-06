from trainer import Trainer, context
import torch
from collections import Counter

__all__ = [
    "VAETrainer"
]
class VAETrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super(VAETrainer, self).__init__(*args, **kwargs)
        self.recons_x = []
        self.latent_dist = []
        self.z = []
    
    def predict(self):
        self.preds = self.model(self.data)
            
    def calc_loss(self):
        self.batch_loss = self.loss_fn(self.data, **self.preds)
        self.loss = self.batch_loss['loss']

    def update_grad(self):
        self.loss.backward()
        self.optim.step()

    def collate_preds(self):
        self.recons_x.append(self.preds['recons_x'])
        self.latent_dist.append(self.preds['latent_dist'])
        self.z.append(self.preds['z'])

    @context('eval')
    def eval(self, epochs=1):
        self.model.infer = True
        for epoch in self.epochs:
            with torch.no_grad():self.one_epoch(False)



import torch
from operator import attrgetter
from callbacks.exceptions import *
from fastprogress import master_bar, progress_bar
from functools import partial
import importlib


__all__ = [
    "context",
    "Trainer"
]

class context:
    def __init__(self, name):
        self.name = name
    def __call__(self, trainer_fn):
        def _f(o, *args, **kwargs):
            try:
                o.callback(f'before_{self.name}')
                trainer_fn(o, *args, **kwargs)
                o.callback(f'after_{self.name}')
            except globals()[f'Cancel{self.name.title()}Exception'] : pass
        return _f
        
class Trainer():
    def __init__(self, 
                 model, 
                 train_dl=None, 
                 valid_dl=None, 
                 test_dl=None, 
                 loss_fn=None, 
                 metric_fn=None, 
                 optim_fn=None,
                 batch_size = None,
                 callbacks=None):
        self.model = model
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        #self.optim_fn = optim_fn
        self.optim = optim_fn
        self.callbacks = callbacks
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl
        self.batch_size = batch_size

        

    @context('batch')
    def _one_batch(self):
        self.data = self.batch[0].to(self.device)
        self.label = self.batch[1].to(self.device)
        self.optim.zero_grad()
        self.predict()
        if self.model.infer:
            self.collate_preds()
        if not self.model.infer:
            self.calc_loss()
        if self.model.training:
            self.update_grad()
                
    @context('epoch')
    def _one_epoch(self):
        for self.i, self.batch in enumerate(self.dl):
            self._one_batch()
            
    def one_epoch(self, task):
        self.model.training = task
        if self.model.training:
            self.dl = self.train_dl
        elif not self.model.infer and not self.model.training:
            self.dl = self.valid_dl
        elif self.model.infer:
            self.dl = self.test_dl
        self.dataset_size = len(self.dl.dataset) 
        self._one_epoch()
        
    @context('fit')
    def _fit(self, train, valid):
        for self.epoch in self.epochs:
            if train: self.one_epoch(True)
            if valid: torch.no_grad()(self.one_epoch)(False)

    def fit(self, epochs=2, train=True, valid=True, start_epoch=0, cbs=None):
        #for cb in [cbs]: self.callbacks.append(cb)
        #try:
        self.epochs = range(start_epoch, epochs)
        self.model.to(self.device)
        self.model.train(train)
        self.model.infer = False
        #self.optim = self.optim_fn
        self._fit(train, valid)
        #finally:
            #for cb in [cbs]: self.callbacks.remove(cb)



    def save(self, name, path=None):
        if not path:
            path = "./chkpts"
        params = {
            'epoch' : self.epoch,
            'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'loss' : self.metrics.stats['train_loss']}
        torch.save(params, f'{path}/{name}.pt')

    def load(self, path=None):
        if not path:
            print("Please specify the path to the torch file.")
        checkpoint = torch.load(path, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']

    def resume(self, epochs, path=None):
        self.load(path)
        last_epoch = self.epoch + 1
        self.fit(epochs = epochs + last_epoch, start_epoch=last_epoch)
                
    def callback(self, name):
        for callback in sorted(self.callbacks, key=attrgetter("order")):
            method = getattr(callback, name)
            if method is not None:
                method(self)
                
    def __getattr__(self, name):
        if name in ['predict', 'calc_loss', 'update_grad', 'collate_preds'] : return partial(self.callback, name)
        raise AttributeError(name)
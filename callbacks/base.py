import math
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ExponentialLR
from torch.distributions.multivariate_normal import MultivariateNormal
from typing import Mapping
from torcheval.metrics import Mean
from fastprogress import master_bar, progress_bar
from copy import copy
from collections import defaultdict
from callbacks.exceptions import *

__all__ = [
    "Callback",
    "DeviceCallback",
    "LRSchedulerCallback",
    "MetricCallback",
    "ProgressCallback"
]

class Callback(): order = 0

class DeviceCallback(Callback):
    def __init__(self):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
            
    def before_fit(self, train):
        train.model = self._to_device(train.model, self.device)

    def before_batch(self, train):
        train.batch = self._to_device(train.batch, self.device)

    def after_batch(self, train):
        train.batch = self._to_device(train.batch, 'cpu')
        train.preds = self._to_device(train.preds, 'cpu')
        if not train.model.infer:
            train.loss = self._to_device(train.loss, 'cpu')

    def _to_device(self, x, device = 'cuda'):
        if isinstance(x, torch.Tensor) : return x.to(device)
        if isinstance(x, nn.Module) : return x.to(device)
        if isinstance(x, Mapping) :
            obj = {}
            for k, v in x.items():
                if isinstance(v, MultivariateNormal) : 
                    loc = v.loc.to(device)
                    cm  = v.covariance_matrix.to(device) # always Covarince Matrix?
                    obj[k] = MultivariateNormal(loc, cm)
                else: obj[k] = v.to(device)
            return obj
        return 'Invalid data/module type.'

    def before_epoch(self, train) : pass
    def after_epoch(self, train) : pass
    def before_batch(self, train) : pass
    def after_fit(self, train) : pass

class LRSchedulerCallback(Callback):
    def __init__(self, sched_fn):
        self.sched_fn = sched_fn

    def before_fit(self, train):
        self.lr_scheduler = self.sched_fn

    def after_batch(self, train):
        if train.model.training : self.lr_scheduler.step()

    def plot(self):
        plt.plot(self.lrs, self.losses)
        plt.xscale('log')

    def after_fit(self, train): pass
    def before_epoch(self, train) : pass
    def after_epoch(self, train) : pass
    def before_batch(self, train) : pass

class MetricCallback(Callback):
    order = Callback.order + 1
    def __init__(self, *metric, **metrics):
        for m in metric:
            metrics[type(m).__name__] = m
        self.metrics = copy(metrics)
        self.metrics['loss'] = Mean()
        self.stats = {}

    def before_fit(self, train):
        train.metrics = self

    def _log(self, stats) : 
        print(stats)
    
    def before_epoch(self, train):
        [m.reset() for m in self.metrics.values()]

    def after_epoch(self, train):            
        log = {f'train_{k}' if train.model.training else f'val_{k}' :f'{v.compute() / train.batch_size:.3f}'  for k, v in self.metrics.items()}
        log['epoch'] = train.epoch
        self.stats.update(log)
        if not train.model.training:
            logs = copy(self.stats)
            self._log(logs)

    def after_batch(self, train):
        if not train.model.infer:
            for k, v in self.metrics.items():
                if k == 'loss': v.update(train.loss)

    def before_batch(self, train):
        pass

    def after_fit(self, train):
        pass

    def before_eval(self, train):
        pass

    def after_eval(self, train):
        pass

class ProgressCallback(Callback):
    
    order = MetricCallback.order + 1
    
    def __init__(self, plot=False):
        self.plot = plot

    def after_fit(self, train):
        pass

    def before_fit(self, train):
        train.epochs = self.mbar = master_bar(train.epochs) # epochs iter type is changed to master bar iter
        self.train_losses = []
        self.val_losses = []
        if hasattr(train, 'metrics') : 
            train.metrics._log = self._log
        self.set_header = True
        self.init_val_loss = True

    def before_eval(self, train):
        train.epochs = self.mbar = master_bar(range(1))

    def after_eval(self, train):
        pass

    def _log(self, data):
        logs = {}
        logs['epoch'] = data.pop('epoch')
        logs['train_loss'] = data.pop('train_loss')
        logs['val_loss'] = data.pop('val_loss')
        [logs.update(d) for d in data]
        if self.set_header: # add column names
            self.mbar.write(list(logs.keys()), table=True)
            self.set_header = False
        self.mbar.write(list(logs.values()), table=True) # add column values

    def before_batch(self, train):
        pass

    def before_epoch(self, train):
        train.dl = progress_bar(train.dl, leave=False, parent=self.mbar) # dataloader type is changed to NBprogress bar

    def after_epoch(self, train):
        if self.plot and hasattr(train, 'metrics'):
            if not train.model.training:
                self.train_losses.append(float(train.metrics.stats['train_loss']))
                self.val_losses.append(float(train.metrics.stats['val_loss']))
                self.mbar.update_graph([[list(range(len(self.train_losses))), self.train_losses], [list(range(len(self.val_losses))), self.val_losses]])
            
    def after_batch(self, train):
        pass
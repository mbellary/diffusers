import torch
import torchvision
import importlib

from torch import tensor
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2


# def dataset_name_mapper(name):
#     dataset_name = {
#         'mnist'         : 'MNIST',
#         'fashion_mnist' : 'FashionMNIST',
#         'cifar10'       : 'CIFAR10'
#     }
#     return dataset_name[name]



__all__ = [
        "load_dataset"
]

def _dataset_stats(name):
    dataset_stats = {
        'MNIST' : ((0.1307,), (0.3081,))
    }
    return dataset_stats[name]

def load_dataset(dataset_name='MNIST', 
                 dataset_path='train_data', 
                 batch_size=1, 
                 n_workers=0,
                 valid_split=0.2,
                keep_channel = True):
    r""" loads a torch dataset

    parameters:
        dataset_name = name of the dataset. This should be a torch dataset
        batch_size = number of samples in each batch.
        shuffle_train = shuffle training samples
        n_workers = Parallel processing

    returns:
        A train, validation and test data loaders.
    """
    seed = torch.Generator().manual_seed(42)
    m = importlib.import_module('torchvision.datasets')
    dataset_fn = getattr(m, dataset_name)
    # mean, std = _dataset_stats(dataset_name)
    transforms = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Lambda(lambda x : x if keep_channel else x.view(-1))
    ])
    
    train_ds = dataset_fn(dataset_path, train=True, download=True, transform=transforms)
    valid_len = int(len(train_ds) * valid_split)
    train_len = int(len(train_ds) - valid_len)
    train_ds, valid_ds = random_split(train_ds, [train_len, valid_len], seed)
    test_ds = dataset_fn(dataset_path, train=False, download=True, transform=transforms)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    
    return train_dl, valid_dl, test_dl
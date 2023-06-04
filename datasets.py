# Code adapted from https://github.com/isl-org/MultiObjectiveOptimization/blob/master/multi_task/

import numpy as np
import random

import torch
from torchvision import transforms
from multi_mnist_loader import MNIST


def global_transformer():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])


def get_dataset(dataset, batch_size, configs, generator=None, worker_init_fn=None, train=True):

    if 'mnist' in dataset:
        if train:
            # Return training + validation split for training loop.
            train_dst = MNIST(root=configs['mnist']['path'], split="train", download=True, transform=global_transformer())
            train_loader = torch.utils.data.DataLoader(train_dst, batch_size=batch_size, shuffle=True, num_workers=2,
                                                       generator=generator, worker_init_fn=worker_init_fn)

            val_dst = MNIST(root=configs['mnist']['path'], split="val", download=True, transform=global_transformer())
            val_loader = torch.utils.data.DataLoader(val_dst, batch_size=100, shuffle=True, num_workers=2,
                                                     generator=generator, worker_init_fn=worker_init_fn)
            return train_loader, val_loader
        else:
            # Return test split only for evaluation of a stored model.
            test_dst = MNIST(root=configs['mnist']['path'], split="test", download=True, transform=global_transformer())
            test_loader = torch.utils.data.DataLoader(test_dst, batch_size=100, shuffle=True, num_workers=2,
                                                      generator=generator, worker_init_fn=worker_init_fn)
            return test_loader
    else:
        raise ValueError(f'the requested data set {dataset} is not implemented')


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


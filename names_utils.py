from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from butd_modules.butd_architectures import get_conv_net, butd_resnet18, BUTDSimpleNet, BUTDTinyResNet, BUTDFCNet
from utils import galu


def name2loss(loss_name):
    if loss_name == 'MSE':
        return nn.MSELoss()
    if loss_name == 'CrossEntropy':
        return nn.CrossEntropyLoss()
    if loss_name == 'BCE':
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"loss {loss_name} is not implemented")


def name2metric(metric_name):
    if metric_name == 'Accuracy':
        return accuracy_score
    if metric_name == 'Micro F1':
        return partial(f1_score, average='micro')
    raise ValueError(f"metric {metric_name} is not implemented")


def name2optim(optimizer_name):
    if optimizer_name in ['SGD', 'sgd']:
        return torch.optim.SGD
    if optimizer_name in ['Adam', 'adam']:
        return torch.optim.Adam
    raise ValueError(f"optimizer {optimizer_name} is not implemented")


def name2non_linear_fn(non_linear_name):
    if non_linear_name in ['relu', 'Relu']:
        return F.relu
    raise ValueError(f"non linear function {non_linear_name} is not implemented")


def name2lateral_fn(lateral_name):
    if lateral_name in ['galu', 'Galu']:
        return galu
    raise ValueError(f"lateral function {lateral_name} is not implemented")


def name2network_module(network_name):
    if network_name == 'simple net':
        return BUTDSimpleNet
    if network_name == 'resnet18':
        return butd_resnet18
    if network_name == 'tiny resnet':
        return BUTDTinyResNet
    if network_name == 'conv net':
        return get_conv_net
    if network_name == 'fc net':
        return BUTDFCNet
    raise ValueError(f"lateral function {network_name} is not implemented")



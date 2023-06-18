import math
import numpy as np
from typing import Union, Tuple, List

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, init


def butd_module_wrapper(cls):
    """
    A wrapper for BUTD layers.
    The class sotres the incoming BU and TD vectors,
    and is charge of applying non-linearity and lateral connections (gating)
    """
    class BUTDModuleWrapper(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.bu_neurons = None
            self.td_neurons = None

            self.non_linear = name2non_linear_fn(kwargs.get('non_linearity_fn', 'relu'))
            self.lateral = name2lateral_fn(kwargs.get('lateral_fn', 'galu'))

        def forward(self, x, non_linear=False, lateral=False, norm_module=None, **kwargs):
            self.bu_neurons = x

            kwargs.update({'non_linear': non_linear, 'lateral': lateral})

            x = super().forward(x, **kwargs)

            if norm_module is not None:
                x = norm_module(x)

            if non_linear:
                x = self.non_linear(x)

            if lateral:
                x = self.lateral(x, self.td_neurons)

            return x

        def back_forward(self, x, non_linear=False, lateral=False, norm_module=None, **kwargs):
            self.td_neurons = x

            kwargs.update({'non_linear': non_linear, 'lateral': lateral})

            x = super().back_forward(x, **kwargs)

            if norm_module is not None:
                x = norm_module.back_forward(x)

            if non_linear:
                x = self.non_linear(x)

            if lateral:
                x = self.lateral(x, self.bu_neurons)

            return x

        @staticmethod
        def _get_name():
            return cls.__name__

    return BUTDModuleWrapper


def galu(x, counter_x):
    return x * (counter_x > 0)
    # return x * (counter_x != 0)


def name2non_linear_fn(non_linear_name):
    if non_linear_name in ['relu', 'Relu', 'ReLU']:
        return F.relu
    raise ValueError(f"non linear function {non_linear_name} is not implemented")


def name2lateral_fn(lateral_name):
    if lateral_name in ['galu', 'Galu', 'GaLU']:
        return galu
    raise ValueError(f"lateral function {lateral_name} is not implemented")


@butd_module_wrapper
class BUTDLinear(nn.Linear):
    """
    A BU-TD implementation of the standard Linear layer.
    It extends the standard layer by adding a back_forward method, and allows Counter-Hebbian learning.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, back_bias: bool = False,
                 shared_weights: bool = True, **kwargs) -> None:
        linear_args = {
            'in_features': in_features,
            'out_features': out_features,
            'bias': bias,
        }
        additional_linear_args = ['device', 'dtype']
        linear_args.update({k: v for k, v in kwargs.items() if k in additional_linear_args})

        super().__init__(**linear_args)

        factory_kwargs = {'device': kwargs.get('device', None), 'dtype': kwargs.get('dtype', None)}

        self.shared_weights = shared_weights

        if shared_weights:
            self.register_parameter('back_weight', None)
        else:
            self.back_weight = Parameter(torch.empty((in_features, out_features), **factory_kwargs))

        if back_bias:
            self.back_bias = Parameter(torch.empty(in_features, **factory_kwargs))
        else:
            self.register_parameter('back_bias', None)

        self.reset_back_parameters()

    def reset_back_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        if not self.shared_weights:
            init.kaiming_uniform_(self.back_weight, a=math.sqrt(5))
        if self.back_bias is not None:
            _, fan_out = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out) if fan_out > 0 else 0
            init.uniform_(self.back_bias, -bound, bound)

    def forward(self, input: Tensor, bias_blocking=False, **kwargs) -> Tensor:
        return F.linear(input, self.weight, self.bias if bias_blocking is False else None)

    def back_forward(self, input: Tensor, bias_blocking=False, **kwargs) -> Tensor:
        if self.shared_weights:
            return F.linear(input, self.weight.T, self.back_bias if bias_blocking is False else None)
        else:
            return F.linear(input, self.back_weight, self.back_bias if bias_blocking is False else None)

    def counter_hebbian_update_value(self, update_forward_bias=True, update_backward_bias=True):
        update_val = torch.matmul(self.td_neurons.unsqueeze(-1), self.bu_neurons.unsqueeze(1)).mean(axis=0)
        if self.weight.grad is None:
            self.weight.grad = update_val
        else:
            self.weight.grad += update_val

        if self.bias is not None and update_forward_bias:
            if self.bias.grad is None:
                self.bias.grad = self.td_neurons.mean(axis=0)
            else:
                self.bias.grad += self.td_neurons.mean(axis=0)

        if self.back_bias is not None and update_backward_bias:
            if self.back_bias.grad is None:
                self.back_bias.grad = self.bu_neurons.mean(axis=0)
            else:
                self.back_bias.grad += self.bu_neurons.mean(axis=0)

        if not self.shared_weights:
            # The TD weights gets the same update as the BU weights (up to a transpose)
            if self.back_weight.grad is None:
                self.back_weight.grad = update_val.T
            else:
                self.back_weight.grad += update_val.T

    def extra_repr(self) -> str:
        s = super().extra_repr()
        return s + ', shared_weights={}'.format(self.shared_weights)


@butd_module_wrapper
class BUTDConv2d(nn.Conv2d):
    """
        A BU-TD implementation of the standard Conv2d layer.
        It extends the standard layer by adding a back_forward method, and allows Counter-Hebbian learning.
        Additionally it can return the spatial shape of an output given the input spatial shape.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], bias: bool = True,
                 back_bias: bool = False, shared_weights: bool = True, **kwargs) -> None:
        conv_args = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'bias': bias,
        }
        additional_conv_args = ['stride', 'padding', 'dilation', 'groups', 'padding_mode', 'device', 'dtype']
        conv_args.update({k: v for k, v in kwargs.items() if k in additional_conv_args})

        super().__init__(**conv_args)

        self.output_padding_ = kwargs.get('output_padding', 0)

        if kwargs.get('output_padding', None) is not None:
            raise ValueError('output padding is not supported for {}'.format(self.__class__.__name__))
        self.shared_weights = shared_weights

        factory_kwargs = {'device': kwargs.get('device', None), 'dtype': kwargs.get('dtype', None)}

        if shared_weights:
            self.register_parameter('back_weight', None)
        else:
            self.back_weight = Parameter(torch.empty(
                (self.out_channels, self.in_channels // self.groups, *self.kernel_size), **factory_kwargs))

        if back_bias:
            self.back_bias = Parameter(torch.empty(self.in_channels, **factory_kwargs))
        else:
            self.register_parameter('back_bias', None)

        self.reset_back_parameters()

    def reset_back_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        if not self.shared_weights:
            init.kaiming_uniform_(self.back_weight, a=math.sqrt(5))
        if self.back_bias is not None:
            _, fan_out = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            init.uniform_(self.back_bias, -bound, bound)

    def forward(self, input: Tensor, bias_blocking=False, **kwargs) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias if bias_blocking is False else None)

    def back_forward(self, input: Tensor, bias_blocking=False, **kwargs) -> Tensor:
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        if self.bu_neurons is not None:
            if self.shared_weights:
                return torch.nn.grad.conv2d_input(self.bu_neurons.shape, self.weight, input, self.stride, self.padding,
                                                  self.dilation, self.groups)
            else:
                return torch.nn.grad.conv2d_input(self.bu_neurons.shape, self.weight, input, self.stride, self.padding,
                                                  self.dilation, self.groups)
        if self.shared_weights:
            return F.conv_transpose2d(
                input, self.weight, self.back_bias if bias_blocking is False else None, self.stride, self.padding,
                kwargs.get('output_padding', self.output_padding_), self.groups, self.dilation)
        else:
            return F.conv_transpose2d(
                input, self.back_weight, self.back_bias if bias_blocking is False else None, self.stride, self.padding,
                0, self.groups, self.dilation)

    def counter_hebbian_update_value(self, update_forward_bias=True, update_backward_bias=True):
        update_val = torch.nn.grad.conv2d_weight(
            self.bu_neurons, self.weight.shape, self.td_neurons,
            stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups
        ) / len(self.bu_neurons)

        if self.weight.grad is None:
            self.weight.grad = update_val
        else:
            self.weight.grad += update_val

        if self.bias is not None and update_forward_bias:
            if self.bias.grad is None:
                self.bias.grad = self.td_neurons.sum(axis=[2, 3]).mean(axis=0)
            else:
                self.bias.grad += self.td_neurons.sum(axis=[2, 3]).mean(axis=0)

        if self.back_bias is not None and update_backward_bias:
            if self.back_bias.grad is None:
                self.back_bias.grad = self.bu_neurons.sum(axis=[2, 3]).mean(axis=0)
            else:
                self.back_bias.grad += self.bu_neurons.sum(axis=[2, 3]).mean(axis=0)

        if not self.shared_weights:
            # The TD weights gets the same update as the BU weights (up to a transpose)
            if self.back_weight.grad is None:
                self.back_weight.grad = update_val
            else:
                self.back_weight.grad += update_val

    def get_out_spatial_shape(self, in_shape):
        out_shape = []
        for dim in range(len(in_shape)):
            out_shape.append(
                int(1 + (in_shape[dim] + 2 * self.padding[dim] - self.dilation[dim] *
                         (self.kernel_size[dim] - 1) - 1) / self.stride[dim])
            )
        return out_shape

    def extra_repr(self) -> str:
        s = super().extra_repr()
        return s + ', shared_weights={}'.format(self.shared_weights)


class BUTDFlatten(nn.Flatten):
    """
        A BU-TD implementation of the standard Flatten layer.
        It extends the standard layer by adding a back_forward method.
    """

    def __init__(self, start_dim: int = 1, end_dim: int = -1, in_shape: List[int] = None) -> None:
        super(BUTDFlatten, self).__init__(start_dim, end_dim)
        if in_shape is None:
            raise ValueError("BUTDFlatten module require in_shape, but None was given")
        self.in_shape = in_shape

    def forward(self, x, **kwargs):
        return super().forward(x)

    def back_forward(self, x, **kwargs):
        return x.reshape([x.shape[0]] + self.in_shape)

    def extra_repr(self) -> str:
        return super().extra_repr() + ' in_shape={}'.format(self.in_shape)

    def counter_hebbian_update_value(self, **kwargs):
        pass


class BUTDSequential(nn.Sequential):
    """
        A BU-TD implementation of the standard Sequential layer.
        It extends the standard layer by adding a back_forward method, and allows Counter-Hebbian learning.
    """
    def forward(self, x, **kwargs):
        for module in self:
            x = module(x, **kwargs)
        return x

    def back_forward(self, x, **kwargs):
        for module in reversed(self):
            x = module.back_forward(x, **kwargs)
        return x

    def counter_hebbian_update_value(self, **kwargs):
        for module in self:
            module.counter_hebbian_update_value(**kwargs)


class BUTDBatchNorm2d(nn.BatchNorm2d):
    """
        A BU-TD implementation of the standard Flatten layer.
        It extends the standard layer by adding a back_forward method.
    """
    def __init__(self, num_features, **kwargs):
        super().__init__(num_features, **kwargs)
        self.back_batch_norm = nn.BatchNorm2d(num_features, **kwargs)

    def forward(self, x, **kwargs):
        return super().forward(x)

    def back_forward(self, x, **kwargs):
        return self.back_batch_norm(x)

    def counter_hebbian_update_value(self, **kwargs):
        pass



from typing import Optional, Callable

from torchvision import models

from custom_layers import *


def butd_conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, **kwargs) -> BUTDConv2d:
    """3x3 convolution with padding"""
    return BUTDConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                      padding=dilation, groups=groups, bias=False, dilation=dilation, **kwargs)


def butd_conv1x1(in_planes: int, out_planes: int, stride: int = 1, **kwargs) -> BUTDConv2d:
    """1x1 convolution"""
    return BUTDConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, **kwargs)


@butd_module_wrapper
class BUTDBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        downdample_norm_layer: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = BUTDBatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        if 'shared_weights' not in kwargs:
            kwargs['shared_weights'] = True
        self.shared_weights = kwargs['shared_weights']

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = butd_conv3x3(inplanes, planes, stride, **kwargs)
        self.conv2 = butd_conv3x3(planes, planes, **kwargs)
        if norm_layer == 'no_norm':
            self.bn1 = None
            self.bn2 = None
        else:
            self.bn1 = norm_layer(planes)
            self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.downdample_norm_layer = downdample_norm_layer
        self.stride = stride

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x, norm_module=self.downdample_norm_layer, non_linear=False, lateral=False,
                                       bias_blocking=kwargs.get('bias_blocking', False))

        out = self.conv1(x, norm_module=self.bn1, **kwargs)

        out = self.conv2(out, norm_module=self.bn2, non_linear=False, lateral=False,
                         bias_blocking=kwargs.get('bias_blocking', False))

        out += identity

        return out

    def back_forward(self, x, **kwargs):
        identity = x

        if self.downsample is not None:
            identity = self.downsample.back_forward(x, norm_module=self.downdample_norm_layer, non_linear=False,
                                                    lateral=False, bias_blocking=kwargs.get('bias_blocking', False))

        out = self.conv2.back_forward(x, norm_module=self.bn2, **kwargs)

        out = self.conv1.back_forward(out, norm_module=self.bn1, non_linear=False, lateral=False,
                                      bias_blocking=kwargs.get('bias_blocking', False))

        out += identity

        return out

    def counter_hebbian_update_value(self, **kwargs):
        self.conv1.counter_hebbian_update_value()
        self.conv2.counter_hebbian_update_value()
        if self.downsample is not None:
            self.downsample.counter_hebbian_update_value()

    def get_out_spatial_shape(self, in_shape):
        if self.downsample is not None:
            return self.downsample.get_out_spatial_shape(in_shape)
        return in_shape


class BUTDBottleneck(nn.Module):
    pass


class BUTDTransformerBlock(nn.Module):
    pass














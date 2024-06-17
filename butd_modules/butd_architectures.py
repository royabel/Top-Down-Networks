from typing import Type, Any

import numpy as np
# from torchvision.models.resnet import load_state_dict_from_url, model_urls

from butd_modules.butd_building_blocks import *


class BUTDConvNet(nn.Module):
    def __init__(self, in_shape: List[int], conv_channels=None, conv_kernels=None, fc_layers=None,
                 shared_weights: bool = True, **kwargs):
        super(BUTDConvNet, self).__init__()
        if (conv_channels is None) + (conv_kernels is None) == 1:
            raise ValueError(f"Expecting both conv_channels and conv_kernels, but only "
                             f"{'conv_channels' * (conv_channels is not None)}"
                             f"{'conv_kernels' * (conv_kernels is not None)} was given")
        if (conv_channels is not None) and (len(conv_channels) != len(conv_kernels)):
            raise ValueError(f"Expecting conv_channels and conv_kernels to be of the same length, "
                             f"but {len(conv_channels)}, {len(conv_kernels)} were given")

        self.in_shape = in_shape
        self.shared_weights = shared_weights

        # Define Convolutional layers
        conv_channels = [] if conv_channels is None else conv_channels
        conv_channels = [in_shape[0]] + conv_channels
        conv_kernels = [] if conv_kernels is None else conv_kernels

        self.layers = nn.ModuleList([
            BUTDConv2d(
                in_channels=conv_channels[i],
                out_channels=conv_channels[i + 1],
                kernel_size=conv_kernels[i],
                shared_weights=shared_weights,
                **kwargs
            )
            for i in range(len(conv_kernels))
        ])

        # Define Fully Connected layers
        fc_layers = [] if fc_layers is None else fc_layers

        spat_shape = None
        if len(in_shape) > 2:
            # Assuming the first dimension is the input channels, and the last two dimensions are the spatial dimensions
            spat_shape = [in_shape[1], in_shape[2]]
            n_channels = in_shape[0]
            for conv_layer in self.layers:
                spat_shape = conv_layer.get_out_spatial_shape(spat_shape)
                n_channels = conv_layer.out_channels
            fc_in_shape = np.prod(spat_shape) * n_channels
        elif len(in_shape) == 1 and len(self.layers) == 0:
            fc_in_shape = in_shape[0]
        else:
            raise NotImplementedError()

        fc_layers = [fc_in_shape] + fc_layers

        self._out_shape = fc_layers[-1]

        if len(fc_layers) > 1 and spat_shape is not None:
            self.layers.append(BUTDFlatten(in_shape=[n_channels] + spat_shape))

        self.layers.extend([
            BUTDLinear(
                in_features=fc_layers[i],
                out_features=fc_layers[i + 1],
                shared_weights=shared_weights,
                **kwargs
            )
            for i in range(len(fc_layers) - 1)
        ])

        # a random run to create bu neurons for each butd module
        with torch.no_grad():
            self.forward(torch.randn(1, *self.in_shape))

    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x

    def back_forward(self, x, **kwargs):
        for layer in reversed(self.layers):
            x = layer.back_forward(x, **kwargs)
        return x

    def counter_hebbian_update_value(self, **kwargs):
        for layer in self.layers:
            layer.counter_hebbian_update_value(**kwargs)

    @property
    def out_shape(self):
        return self._out_shape


def get_conv_net(in_shape, shared_weights, **kwargs):
    kwargs.update({
        'conv_channels': [32]*4,
        'conv_kernels': [5]*4,
        'fc_layers': [100],
    })
    return BUTDConvNet(in_shape=in_shape, shared_weights=shared_weights, **kwargs)


class BUTDResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BUTDBasicBlock, BUTDBottleneck]],
        layers: List[int],
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        in_shape: List[int] = None,
        shared_weights: bool = True,
    ) -> None:
        super(BUTDResNet, self).__init__()
        if in_shape is None:
            raise ValueError(f"must give in_shape, but None was given")
        if in_shape[0] != 3:
            raise ValueError(f"support only inputs with shape 3xHxW, but {in_shape[0]} channels were given")

        self.shared_weights = shared_weights
        self.in_shape = in_shape

        if norm_layer is None:
            norm_layer = BUTDBatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = BUTDConv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False,
                                shared_weights=self.shared_weights)
        if norm_layer == 'no_norm':
            self.bn1 = None
        else:
            self.bn1 = norm_layer(self.inplanes, back_num_features=self.in_shape[0])

        channels = [64, 128, 256, 512]

        self.layer1 = self._make_layer(block, channels[0], layers[0])
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        out_spatial_shape = self.conv1.get_out_spatial_shape(in_shape[1:])
        out_spatial_shape = self.layer1[0].conv1.get_out_spatial_shape(out_spatial_shape)
        out_spatial_shape = self.layer2[0].conv1.get_out_spatial_shape(out_spatial_shape)
        out_spatial_shape = self.layer3[0].conv1.get_out_spatial_shape(out_spatial_shape)
        out_spatial_shape = self.layer4[0].conv1.get_out_spatial_shape(out_spatial_shape)
        out_shape = [channels[3]] + out_spatial_shape

        # self.avgpool = BUTDSpatialAvgPool2d()
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.flatten = BUTDFlatten(in_shape=out_shape)
        self._out_shape = np.prod(out_shape)

        self.layers = BUTDSequential(self.layer1, self.layer2, self.layer3, self.layer4, self.flatten)

        # a random run to create bu neurons for each butd module
        with torch.no_grad():
            self.forward(torch.randn(1, *self.in_shape))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[Union[BUTDBasicBlock, BUTDBottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> BUTDSequential:
        norm_layer = self._norm_layer
        downsample = None
        downsample_norm_layer = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = butd_conv1x1(self.inplanes, planes * block.expansion, stride)
            if norm_layer != 'no_norm':
                downsample_norm_layer = norm_layer(planes * block.expansion, back_num_features=self.inplanes)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, downsample_norm_layer, self.groups,
                            self.base_width, previous_dilation, norm_layer, shared_weights=self.shared_weights))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, shared_weights=self.shared_weights))

        return BUTDSequential(*layers)

    def _forward_impl(self, x: Tensor, **kwargs) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x, norm_module=self.bn1, **kwargs)

        x = self.layers(x, **kwargs)

        return x

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self._forward_impl(x, **kwargs)

    def back_forward(self, x, **kwargs):
        x = self.layers.back_forward(x, **kwargs)

        x = self.conv1.back_forward(x, norm_module=self.bn1, **kwargs)
        return x

    def counter_hebbian_update_value(self, **kwargs):
        self.conv1.counter_hebbian_update_value(**kwargs)
        self.layers.counter_hebbian_update_value(**kwargs)

    @property
    def out_shape(self):
        return self._out_shape


def _butd_resnet(
    arch: str,
    block: Type[Union[BUTDBasicBlock, BUTDBottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> BUTDResNet:
    resnet_args = ['groups', 'width_per_group', 'replace_stride_with_dilation', 'norm_layer', 'in_shape', 'shared_weights']
    kwargs = {k: v for k, v in kwargs.items() if k in resnet_args}
    model = BUTDResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError("pre-trained resnet is not supported yet")
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        # model.load_state_dict(state_dict)
    return model


def butd_resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> BUTDResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _butd_resnet('resnet18', BUTDBasicBlock, [2, 2, 2, 2], pretrained, progress,
                        **kwargs)



class BUTDSimpleNet(nn.Module):
    def __init__(self, in_shape: List[int], shared_weights: bool = True, **kwargs):
        super(BUTDSimpleNet, self).__init__()

        self.in_shape = in_shape
        self.shared_weights = shared_weights

        if kwargs.get('conv_channels', None) is not None:
            conv1_channels = kwargs['conv_channels'][0]
            conv2_channels = kwargs['conv_channels'][1]
        else:
            conv1_channels = 32
            conv2_channels = 32
        out_shape = kwargs.get('last_hidden_size', 50)

        self._conv1 = BUTDConv2d(in_shape[0], conv1_channels, kernel_size=5, shared_weights=shared_weights, **kwargs)
        self._conv1_pool = BUTDConv2d(conv1_channels, conv1_channels, kernel_size=2, stride=2,
                                      groups=conv1_channels, shared_weights=shared_weights, **kwargs)

        self._conv2 = BUTDConv2d(conv1_channels, conv2_channels, kernel_size=5, shared_weights=shared_weights, **kwargs)
        self._conv2_pool = BUTDConv2d(conv2_channels, conv2_channels, kernel_size=3, stride=2,
                                      groups=conv2_channels, shared_weights=shared_weights, **kwargs)

        spat_shape = None
        if len(in_shape) == 3:
            # Assuming the first dimension is the input channels, and the last two dimensions are the spatial dimensions
            spat_shape = [in_shape[1], in_shape[2]]
            spat_shape = self._conv1.get_out_spatial_shape(spat_shape)
            spat_shape = self._conv1_pool.get_out_spatial_shape(spat_shape)
            spat_shape = self._conv2.get_out_spatial_shape(spat_shape)
            spat_shape = self._conv2_pool.get_out_spatial_shape(spat_shape)

            fc_in_shape = np.prod(spat_shape) * conv2_channels
        else:
            raise ValueError(f"in shape should be 3 dimensional: CxHxW, but {len(in_shape)} were dimensions were given")

        self._flat = BUTDFlatten(in_shape=[conv2_channels] + spat_shape)
        self._fc = BUTDLinear(fc_in_shape, out_shape, shared_weights=shared_weights, **kwargs)

        self._out_shape = out_shape

        self.layers = BUTDSequential(self._conv1, self._conv1_pool, self._conv2, self._conv2_pool, self._flat, self._fc)

        # a random run to create bu neurons for each butd module
        with torch.no_grad():
            self.forward(torch.randn(1, *self.in_shape))

    def forward(self, x, **kwargs):
        x = self.layers(x, **kwargs)
        return x

    def back_forward(self, x, **kwargs):
        x = self.layers.back_forward(x, **kwargs)
        return x

    def counter_hebbian_update_value(self, **kwargs):
        for layer in self.layers:
            layer.counter_hebbian_update_value(**kwargs)

    @property
    def out_shape(self):
        return self._out_shape


class BUTDTinyResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BUTDBasicBlock, BUTDBottleneck]]=None,
        layers: List[int]=None,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = 'no_norm',
        in_shape: List[int] = None,
        shared_weights: bool = True,
        **kwargs,
    ) -> None:
        super(BUTDTinyResNet, self).__init__()
        if in_shape is None:
            raise ValueError(f"must give in_shape, but None was given")
        # if in_shape[0] != 3:
        #     raise ValueError(f"support only inputs with shape 3xHxW, but {in_shape[0]} channels were given")
        if layers is None:
            layers = [1, 1]
        if block is None:
            block = BUTDBasicBlock

        self.shared_weights = shared_weights
        self.in_shape = in_shape

        if norm_layer is None:
            norm_layer = BUTDBatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = BUTDConv2d(in_shape[0], self.inplanes, kernel_size=3, stride=1, padding=1, bias=False,
                                shared_weights=self.shared_weights)
        if norm_layer == 'no_norm':
            self.bn1 = None
        else:
            self.bn1 = norm_layer(self.inplanes)

        channels = [64, 64]

        self.layer1 = self._make_layer(block, channels[0], layers[0])
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])

        self.conv_pooling = BUTDConv2d(channels[-1], channels[-1], kernel_size=5,
                                       stride=3, padding=1, shared_weights=self.shared_weights)

        out_spatial_shape = self.conv1.get_out_spatial_shape(in_shape[1:])
        out_spatial_shape = self.layer1[0].conv1.get_out_spatial_shape(out_spatial_shape)
        out_spatial_shape = self.layer2[0].conv1.get_out_spatial_shape(out_spatial_shape)
        out_spatial_shape = self.conv_pooling.get_out_spatial_shape(out_spatial_shape)
        out_shape = [channels[1]] + out_spatial_shape

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.flatten = BUTDFlatten(in_shape=out_shape)
        self._out_shape = np.prod(out_shape)

        self.layers = BUTDSequential(self.layer1, self.layer2, self.conv_pooling, self.flatten)

        # a random run to create bu neurons for each butd module
        with torch.no_grad():
            self.forward(torch.randn(1, *self.in_shape))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[Union[BUTDBasicBlock, BUTDBottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> BUTDSequential:
        norm_layer = self._norm_layer
        downsample = None
        downsample_norm_layer = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = butd_conv1x1(self.inplanes, planes * block.expansion, stride)
            if norm_layer != 'no_norm':
                downsample_norm_layer = norm_layer(planes * block.expansion)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, downsample_norm_layer, self.groups,
                            self.base_width, previous_dilation, norm_layer, shared_weights=self.shared_weights))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, shared_weights=self.shared_weights))

        return BUTDSequential(*layers)

    def _forward_impl(self, x: Tensor, **kwargs) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x, norm_module=self.bn1, **kwargs)

        x = self.layers(x, **kwargs)

        return x

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self._forward_impl(x, **kwargs)

    def back_forward(self, x, **kwargs):
        x = self.layers.back_forward(x, **kwargs)

        x = self.conv1.back_forward(x, norm_module=self.bn1, **kwargs)
        return x

    def counter_hebbian_update_value(self, **kwargs):
        self.conv1.counter_hebbian_update_value(**kwargs)
        self.layers.counter_hebbian_update_value(**kwargs)

    @property
    def out_shape(self):
        return self._out_shape


# A fully connected network with 1 hidden layer
class BUTDFCNet(nn.Module):
    def __init__(self, in_shape: List[int], shared_weights: bool = True, **kwargs):
        super(BUTDFCNet, self).__init__()

        self.in_shape = list(in_shape)
        self.shared_weights = shared_weights

        self._out_shape = kwargs.get('last_hidden_size', 500)

        fc_in_shape = np.prod(self.in_shape)

        self._flat = BUTDFlatten(in_shape=self.in_shape)

        self._fc = BUTDLinear(fc_in_shape, self._out_shape, shared_weights=shared_weights, **kwargs)

        self.layers = BUTDSequential(self._flat, self._fc)

        # a random run to create bu neurons for each butd module
        with torch.no_grad():
            self.forward(torch.randn(1, *self.in_shape))

    def forward(self, x, **kwargs):
        x = self.layers(x, **kwargs)
        return x

    def back_forward(self, x, **kwargs):
        x = self.layers.back_forward(x, **kwargs)
        return x

    def counter_hebbian_update_value(self, **kwargs):
        for layer in self.layers:
            layer.counter_hebbian_update_value(**kwargs)

    @property
    def out_shape(self):
        return self._out_shape






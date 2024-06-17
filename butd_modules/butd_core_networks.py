from butd_modules.butd_architectures import *


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

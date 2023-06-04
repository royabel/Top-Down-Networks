"""
Implements the inputs, custom BUTD layers, BUTD modules
"""

import pytest

from butd_network_building_blocks import *
from custom_layers import *
from core_networks import *

SharedWeights = True

InShape1D = 100
InShape3D = [3, 28, 28]
BatchSize = 32
Bias = True


@pytest.fixture
def my_1d_input():
    return torch.randn(BatchSize, InShape1D)


@pytest.fixture
def my_3d_input():
    return torch.randn(BatchSize, *InShape3D)


@pytest.fixture
def inputs_tensors(my_1d_input, my_3d_input):
    return {
        '1d_input': my_1d_input,
        '3d_input': my_3d_input,
    }


@pytest.fixture
def butd_linear_layer():
    layer_args = {
        'in_features': InShape1D,
        'out_features': 10,
        'bias': Bias,
    }

    butd_layer = BUTDLinear(shared_weigts=SharedWeights, **layer_args)
    origin_layer = nn.Linear(**layer_args)
    origin_layer.weight = butd_layer.weight
    origin_layer.bias = butd_layer.bias

    # The classification BUTDNet requires out_shape attribute
    butd_layer.out_shape = butd_layer.out_features

    return butd_layer, origin_layer


@pytest.fixture
def butd_conv_layer():
    layer_args = {
        'in_channels': InShape3D[0],
        'out_channels': 3,
        'kernel_size': 3,
        'bias': Bias,
        'stride': 2,
        'groups': 3,
    }

    butd_layer = BUTDConv2d(shared_weigts=SharedWeights, **layer_args)
    conv_out_shape = [layer_args['out_channels']] + butd_layer.get_out_spatial_shape(InShape3D[1:])
    butd_module = BUTDSequential(butd_layer, BUTDFlatten(in_shape=conv_out_shape))

    butd_module.out_shape = np.prod(conv_out_shape)
    butd_module.shared_weights = SharedWeights

    origin_layer = nn.Conv2d(**layer_args)
    origin_layer.weight = butd_layer.weight
    origin_layer.bias = butd_layer.bias
    origin_module = nn.Sequential(origin_layer, nn.Flatten())

    return butd_module, origin_module


@pytest.fixture
def butd_conv_max_pool_layer():
    conv_layer_args = {
        'in_channels': InShape3D[0],
        'out_channels': 3,
        'kernel_size': 3,
        'bias': Bias,
    }

    max_pool_layer_args = {
        'kernel_size': 2,
        'stride': 2,
    }

    butd_conv_l = BUTDConv2d(shared_weigts=SharedWeights, **conv_layer_args)
    conv_out_shape = [conv_layer_args['out_channels']] + butd_conv_l.get_out_spatial_shape(InShape3D[1:])
    butd_max_pool_l = BUTDMaxPool2D(**max_pool_layer_args)
    max_pool_out_shape = [conv_out_shape[0]] + butd_max_pool_l.get_out_spatial_shape(conv_out_shape[1:])
    butd_module = BUTDSequential(butd_conv_l, butd_max_pool_l, BUTDFlatten(in_shape=max_pool_out_shape))

    butd_module.out_shape = np.prod(max_pool_out_shape)
    butd_module.shared_weights = SharedWeights

    origin_conv_l = nn.Conv2d(**conv_layer_args)
    origin_conv_l.weight = butd_conv_l.weight
    origin_conv_l.bias = butd_conv_l.bias
    origin_max_pool_l = nn.MaxPool2d(**max_pool_layer_args)
    origin_module = nn.Sequential(origin_conv_l, origin_max_pool_l, nn.Flatten())

    return butd_module, origin_module


@pytest.fixture
def butd_conv_spatial_avg_pool_layer():
    conv_layer_args = {
        'in_channels': InShape3D[0],
        'out_channels': 3,
        'kernel_size': 3,
        'bias': Bias,
    }

    butd_conv_l = BUTDConv2d(shared_weigts=SharedWeights, **conv_layer_args)
    butd_spatial_avg_pool_l = BUTDSpatialAvgPool2d()
    butd_module = BUTDSequential(butd_conv_l, butd_spatial_avg_pool_l)

    butd_module.out_shape = conv_layer_args['out_channels']
    butd_module.shared_weights = SharedWeights

    origin_conv_l = nn.Conv2d(**conv_layer_args)
    origin_conv_l.weight = butd_conv_l.weight
    origin_conv_l.bias = butd_conv_l.bias
    origin_spatial_avg_pool_l = nn.AdaptiveAvgPool2d((1, 1))
    origin_module = nn.Sequential(origin_conv_l, origin_spatial_avg_pool_l, nn.Flatten())

    return butd_module, origin_module


@pytest.fixture
def butd_conv_batch_norm_layer():
    conv_layer_args = {
        'in_channels': InShape3D[0],
        'out_channels': 3,
        'kernel_size': 3,
        'bias': Bias,
    }

    butd_conv_l = BUTDConv2d(shared_weigts=SharedWeights, **conv_layer_args)
    conv_out_shape = [conv_layer_args['out_channels']] + butd_conv_l.get_out_spatial_shape(InShape3D[1:])
    butd_batch_norm_l = BUTDBatchNorm2d(num_features=conv_layer_args['out_channels'])
    butd_module = BUTDSequential(butd_conv_l, butd_batch_norm_l, BUTDFlatten(in_shape=conv_out_shape))

    butd_module.out_shape = np.prod(conv_out_shape)
    butd_module.shared_weights = SharedWeights

    origin_conv_l = nn.Conv2d(**conv_layer_args)
    origin_conv_l.weight = butd_conv_l.weight
    origin_conv_l.bias = butd_conv_l.bias
    origin_batch_norm_l = nn.BatchNorm2d(num_features=conv_layer_args['out_channels'])
    origin_module = nn.Sequential(origin_conv_l, origin_batch_norm_l, nn.Flatten())

    return butd_module, origin_module


@pytest.fixture
def butd_layers(butd_linear_layer, butd_conv_layer, butd_conv_spatial_avg_pool_layer):
    return {
        'linear': butd_linear_layer,
        'conv': butd_conv_layer,
        'conv spatial avg pool': butd_conv_spatial_avg_pool_layer,
        # 'conv max pool': butd_conv_max_pool_layer,
        # 'conv batch norm': butd_conv_batch_norm_layer,
    }


@pytest.fixture
def butd_conv_net():
    module_args = {
        'in_shape': InShape3D,
        'conv_kernels': [3] * 3,
        'conv_channels': [3] * 3,
        'fc_layers': [200, 100],
        'shared_weights': SharedWeights,
        'bias': Bias,
    }

    butd_module = BUTDConvNet(**module_args)

    return butd_module


@pytest.fixture
def butd_simple_net():
    module_args = {
        'in_shape': InShape3D,
        'shared_weights': SharedWeights,
    }

    butd_module = BUTDSimpleNet(**module_args)

    return butd_module


@pytest.fixture
def butd_resnet_block():
    module_args = {
        'inplanes': InShape3D[0],
        'planes': InShape3D[0],
        'norm_layer': 'no_norm',
        'shared_weights': SharedWeights,
    }

    butd_resnet_block_module1 = BUTDBasicBlock(**module_args)
    butd_resnet_block_module2 = BUTDBasicBlock(**module_args)
    module_out_shape = [module_args['planes']] + InShape3D[1:]
    butd_module = BUTDSequential(
        butd_resnet_block_module1,
        butd_resnet_block_module2,
        BUTDFlatten(in_shape=module_out_shape)
    )

    butd_module.out_shape = np.prod(module_out_shape)
    butd_module.shared_weights = SharedWeights

    return butd_module


@pytest.fixture
def butd_resnet():
    module_args = {
        'in_shape': InShape3D,
        'norm_layer': 'no_norm',
    }

    butd_module = butd_resnet18(**module_args)

    return butd_module


@pytest.fixture
def butd_networks(butd_conv_net, butd_resnet_block, butd_resnet, butd_simple_net):
    return {
        'conv_net': butd_conv_net,
        'resnet_block': butd_resnet_block,
        'resnet': butd_resnet,
        'simple net': butd_simple_net,
    }




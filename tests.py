"""
1. Checking that my forward aligns with the standard Pytorch forward for different layers
2. Verifying that my gradients calculation aligns with the real gradients (classification)
3. Verifying task guidance gradients
"""

import pytest

from names_utils import name2loss
from network_models import *
from utils import loss_grads

LossName = 'CrossEntropy'
LossFn = name2loss(LossName)

NClasses = 10


@pytest.mark.parametrize("input_arg,layer_arg", [
    ('1d_input', 'linear'),
    ('3d_input', 'conv'),
    # ('3d_input', 'conv spatial avg pool'),
    # ('3d_input', 'conv max pool'),
    # ('3d_input', 'conv batch norm'),
])
class TestBUTDLayer:
    def test_forward(self, inputs_tensors, butd_layers, layer_arg, input_arg):
        butd_layer, origin_layer = butd_layers[layer_arg]
        x = inputs_tensors[input_arg]

        origin_y = origin_layer(x)
        butd_y = butd_layer(x)

        # Compare the BUTD layer output to the original layer output
        assert origin_y.shape == butd_y.shape
        assert (origin_y == butd_y).all()

    def test_backward(self, inputs_tensors, butd_layers, layer_arg, input_arg):
        butd_layer, _ = butd_layers[layer_arg]
        x = inputs_tensors[input_arg]

        _test_backward(butd_layer, x)

    def test_training(self, inputs_tensors, butd_layers, layer_arg, input_arg):
        butd_layer, _ = butd_layers[layer_arg]
        x = inputs_tensors[input_arg]

        _test_training(butd_layer, x)


@pytest.mark.parametrize("input_arg,network_arg", [
    ('3d_input', 'conv_net'),
    ('3d_input', 'resnet_block'),
    ('3d_input', 'resnet'),
    ('3d_input', 'simple net'),
])
class TestBUTDModule:
    def test_backward(self, inputs_tensors, butd_networks, network_arg, input_arg):
        butd_net = butd_networks[network_arg]
        x = inputs_tensors[input_arg]

        _test_backward(butd_net, x)

    def test_training(self, inputs_tensors, butd_networks, network_arg, input_arg):
        butd_net = butd_networks[network_arg]
        x = inputs_tensors[input_arg]

        _test_training(butd_net, x)

    def test_mtl(self, inputs_tensors, butd_networks, network_arg, input_arg):
        pass


class TestCHLoss:
    pass


def _test_backward(butd_module, x):
    net = ClassificationBUTDNet(n_classes=NClasses, shared_weights=butd_module.shared_weights,
                                core_network=butd_module)
    net.train()

    _single_update_test(x, net)


def _test_training(butd_module, x):
    net = ClassificationBUTDNet(n_classes=NClasses, shared_weights=butd_module.shared_weights,
                                core_network=butd_module)
    net.train()

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    for i in range(10):
        optimizer.zero_grad()
        _single_update_test(x, net)
        optimizer.step()


def _single_update_test(x, network):
    y = network(torch.randn_like(x))

    gt = torch.zeros_like(y)
    gt += F.one_hot(torch.randint(gt.shape[1], size=(gt.shape[0],)), gt.shape[-1])
    if LossName in ['BCE', 'MSE']:
        gt += F.one_hot(torch.randint(gt.shape[1], size=(gt.shape[0],)), gt.shape[-1])
        if LossName == 'BCE':
            gt = torch.clamp(gt, 0, 1)

    loss = LossFn(y, gt)
    loss.backward()

    bp_grads = []
    for param in network.parameters():
        bp_grads.append(param.grad.clone())
        param.grad.zero_()

    # Counter-Hebbian Back-Prop
    d_l_d_outputs = loss_grads(y, gt, loss_name=LossName, n_classes=gt.shape[-1])
    network.counter_hebbian_back_prop(d_l_d_outputs)

    for i, param in enumerate(network.parameters()):
        assert compare_gradients(bp_grads[i], param.grad)


def compare_gradients(grads1, grads2):
    return (((((grads1 - grads2).abs() / (grads1.abs() + 1e-9)) < 5e-2) * (grads1.abs() >= 1e-6)) +
            (((grads1 - grads2).abs() < 1e-8) * (grads1.abs() < 1e-6))).all()

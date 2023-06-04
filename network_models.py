from custom_layers import *


class ClassificationBUTDNet(nn.Module):
    def __init__(self, n_classes: int, shared_weights, core_network, **kwargs):
        super(ClassificationBUTDNet, self).__init__()
        self.shared_weights = shared_weights
        self.n_classes = n_classes
        self.core_net = core_network

        self.head_layer = BUTDLinear(in_features=self.core_net.out_shape, out_features=n_classes,
                                     shared_weights=shared_weights, **kwargs)

    def forward(self, x: Tensor, non_linear=True, lateral=False,
                head_non_linear=False, head_lateral=False, **kwargs):
        x = self.core_net(x, non_linear=non_linear, lateral=lateral, **kwargs)

        x = self.head_layer(x, non_linear=head_non_linear, lateral=head_lateral, **kwargs)

        return x

    def back_forward(self, x: Tensor, non_linear=False, lateral=True,
                     head_non_linear=False, head_lateral=True, **kwargs):
        x = self.head_layer.back_forward(x, non_linear=head_non_linear, lateral=head_lateral, **kwargs)

        x = self.core_net.back_forward(x, non_linear=non_linear, lateral=lateral, **kwargs)

        return x

    def counter_hebbian_back_prop(self, grads, **kwargs):
        # propagate gradients (back propagation)
        self.back_forward(grads, non_linear=False, lateral=True, head_non_linear=False, head_lateral=True,
                          bias_blocking=True, **kwargs)

        self.counter_hebbian_update_value(**kwargs)

    def counter_hebbian_update_value(self, update_forward_bias=True, update_backward_bias=True):
        self.core_net.counter_hebbian_update_value(
            update_forward_bias=update_forward_bias, update_backward_bias=update_backward_bias)
        self.head_layer.counter_hebbian_update_value(
            update_forward_bias=update_forward_bias, update_backward_bias=update_backward_bias)

    def task_guidance_forward(self, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def probs(network_outputs):
        return F.softmax(network_outputs, axis=1)

    @staticmethod
    def multi_label_probs(network_outputs):
        return torch.sigmoid(network_outputs)

    @staticmethod
    def predict(network_outputs):
        return torch.argmax(network_outputs, axis=1)


class TaskBUTDNet(ClassificationBUTDNet):
    def __init__(self, task_size, task_embedding_size=None, **kwargs):
        super().__init__(**kwargs)

        self.task_size = task_size

        task_layers = [self.core_net.out_shape]
        if task_embedding_size is not None:
            task_layers.append(task_embedding_size)
        task_layers.append(task_size)

        self.task_head = BUTDSequential(*[
            BUTDLinear(in_features=task_layers[i], out_features=task_layers[i+1], **kwargs)
            for i in range(len(task_layers)-1)
        ])

    def forward(self, x: Tensor, non_linear=True, lateral=False, task_head=False,
                head_non_linear=False, head_lateral=False, **kwargs):
        x = self.core_net(x, non_linear=non_linear, lateral=lateral, **kwargs)

        if task_head:
            x = self.task_head(x, non_linear=head_non_linear, lateral=head_lateral, **kwargs)
        else:
            x = self.head_layer(x, non_linear=head_non_linear, lateral=head_lateral, **kwargs)

        return x

    def back_forward(self, x: Tensor, non_linear=False, lateral=True, task_head=False,
                     head_non_linear=False, head_lateral=True, **kwargs):
        if task_head:
            x = self.task_head.back_forward(x, non_linear=head_non_linear, lateral=head_lateral, **kwargs)
        else:
            x = self.head_layer.back_forward(x, non_linear=head_non_linear, lateral=head_lateral, **kwargs)

        x = self.core_net.back_forward(x, non_linear=non_linear, lateral=lateral, **kwargs)

        return x

    def task_guidance_forward(self, x, task, **kwargs):
        td_out = self.back_forward(task, non_linear=True, lateral=False, task_head=True,
                                   head_non_linear=True, head_lateral=False)

        # x = self.forward(x + td_out, non_linear=True, lateral=True, task_head=False, **kwargs)
        x = self.forward(x, non_linear=True, lateral=True, task_head=False, **kwargs)

        return x








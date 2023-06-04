from butd_modules.butd_layers import *


class ClassificationBUTDNet(nn.Module):
    def __init__(self, n_classes: int, shared_weights, core_network, **kwargs):
        super(ClassificationBUTDNet, self).__init__()
        self.shared_weights = shared_weights
        self.n_classes = n_classes
        self.core_net = core_network

        self.multi_decoders = False

        self.head_layer = BUTDLinear(in_features=self.core_net.out_shape, out_features=n_classes,
                                     shared_weights=shared_weights, **kwargs)

    def _forward_impl(self, x: Tensor, non_linear=True, lateral=False,
                      head_non_linear=False, head_lateral=False, **kwargs):
        x = self.core_net(x, non_linear=non_linear, lateral=lateral, **kwargs)

        x = self.head_layer(x, non_linear=head_non_linear, lateral=head_lateral, **kwargs)

        return x

    def forward(self, x: Tensor, non_linear=True, lateral=False,
                head_non_linear=False, head_lateral=False, **kwargs):
        return self._forward_impl(x, non_linear=non_linear, lateral=lateral,
                                  head_non_linear=head_non_linear, head_lateral=head_lateral, **kwargs)

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

    @staticmethod
    def probs(network_outputs):
        return F.softmax(network_outputs, axis=1)

    @staticmethod
    def multi_label_probs(network_outputs):
        return torch.sigmoid(network_outputs)

    @staticmethod
    def predict(network_outputs):
        if len(network_outputs.shape) == 1 or (len(network_outputs.shape) == 2 and network_outputs.shape[1] == 1):
            if len(network_outputs.shape) == 2:
                network_outputs = network_outputs[:, 0]
            return (network_outputs > 0).int()
        return torch.argmax(network_outputs, axis=1)


class TaskBUTDNet(ClassificationBUTDNet):
    def __init__(self, task_vector_size, n_tasks, task_embedding_size=None, **kwargs):
        super().__init__(**kwargs)

        self.task_vector_size = task_vector_size
        self.n_tasks = n_tasks
        self.multi_decoders = kwargs.get('multi_decoders', False)

        if self.multi_decoders:
            self.head_layer = BUTDLinear(in_features=self.core_net.out_shape, out_features=self.n_classes * self.n_tasks,
                                         **kwargs)

        task_layers = [self.core_net.out_shape]
        if task_embedding_size is not None:
            task_layers.append(task_embedding_size)
        task_layers.append(task_vector_size)

        self.task_head = BUTDSequential(*[
            BUTDLinear(in_features=task_layers[i], out_features=task_layers[i+1], **kwargs)
            for i in range(len(task_layers)-1)
        ])

    def _forward_impl(self, x: Tensor, non_linear=True, lateral=False, task_head=False,
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

    def forward(self, x: Tensor, task=None, non_linear=True, lateral=False, task_head=False,
                head_non_linear=False, head_lateral=False, pure_bu=False, **kwargs):
        if task is None:
            return self._forward_impl(x, non_linear=non_linear, lateral=lateral, task_head=task_head,
                                      head_non_linear=head_non_linear, head_lateral=head_lateral, **kwargs)

        # Task guidance forward. It includes two steps:
        # 1) a backward pass with task as input to select the task-dependent sub-network
        # 2) a forward pass made on the selected sub-network

        with torch.no_grad():
            self.back_forward(task, non_linear=True, lateral=False, task_head=True,
                              head_non_linear=True, head_lateral=False)

        x = self._forward_impl(x, non_linear=True, lateral=True, task_head=False, **kwargs)

        if self.multi_decoders:
            if self.n_classes > 1:
                x = x.reshape(x.shape[0], -1, self.n_classes)

            # Select the output neurons correspond with the requested task
            x = x[task == 1]

        return x








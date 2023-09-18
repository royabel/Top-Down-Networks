import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import utils


def galu(x, counter_x):
    return x * (counter_x > 0)
    # return x * (counter_x != 0)


def update_in_shape_and_n_classes(params, data_loader):
    # auto adjust data set input size and number of classes
    data_sample = data_loader.dataset[0]

    if hasattr(data_loader.dataset, 'n_classes'):
        params['n_classes'] = data_loader.dataset.n_classes
    elif hasattr(data_loader.dataset, 'classes'):
        params['n_classes'] = len(data_loader.dataset.classes)
    elif len(data_sample[1].shape) > 1:
        params['n_classes'] = data_sample[1].shape[-1]
    elif hasattr(data_loader.dataset, 'train_labels_l'):
        params['n_classes'] = data_loader.dataset.train_labels_l.max() + 1
    else:
        raise ValueError('Cannot infer the number of classes from the data loader object')

    if isinstance(data_sample[0], list):
        in_sample_ = data_sample[0][0]
    else:
        in_sample_ = data_sample[0]

    params['in_shape'] = in_sample_.shape

    return params


def show_images_batch_tensor(images_batch):
    """Show images for a batch of samples."""
    images_batch = images_batch.detach().cpu()

    grid = utils.make_grid(images_batch, pad_value=0.5)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.show()


def loss_grads(outputs, labels, loss_name=None, n_classes=2):
    loss_name = 'BCE' if loss_name is None else loss_name
    if loss_name == 'BCE':
        # BCE gradients
        return (labels * (torch.sigmoid(outputs) - 1) + (1 - labels) * torch.sigmoid(outputs)) / labels.shape[-1]
    if loss_name == 'CrossEntropy':
        if len(labels.shape) > 1:
            return F.softmax(outputs, dim=1) - labels
        return F.softmax(outputs, dim=1) - F.one_hot(labels, n_classes)
    if loss_name == 'MSE':
        # MSE gradients
        return (2 / n_classes) * (outputs - labels)
    raise ValueError(f"loss {loss_name} was not implemented")



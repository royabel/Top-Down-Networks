from torchvision import datasets
import torchvision.transforms as transforms

import data.multi_mnist_loader as multi_mnist_loader


def get_train_n_test_datasets(dataset_name, data_path='./Datasets', **kwargs):
    if dataset_name == 'mnist':
        train_dataset = datasets.MNIST(data_path, train=True, download=True,
                                       transform=global_transformer())
        test_dataset = datasets.MNIST(data_path, train=False,
                                      transform=global_transformer())
    elif dataset_name == 'fashion_mnist':
        train_dataset = datasets.FashionMNIST(data_path, train=True, download=True,
                                              transform=global_transformer())
        test_dataset = datasets.FashionMNIST(data_path, train=False,
                                             transform=global_transformer())
    elif dataset_name == 'cifar':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
            # transforms.Lambda(lambda x: x.to(Device))
        )
        train_dataset = datasets.CIFAR10(data_path, train=True, download=True,
                                         transform=transform)
        test_dataset = datasets.CIFAR10(data_path, train=False,
                                        transform=transform)
    elif dataset_name == 'celeba':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # transforms.Lambda(lambda x: x.to(Device))
             ],
        )
        train_dataset = datasets.CelebA(data_path, split='train', download=False,
                                        transform=transform)
        test_dataset = datasets.CelebA(data_path, split='test', download=False,
                                       transform=transform)

        train_dataset.n_classes = 1
        test_dataset.n_classes = 1
        # train_dataset.n_classes = 40
        # test_dataset.n_classes = 40
    elif dataset_name == 'multi_mnist':
        train_dataset = multi_mnist_loader.MNIST(root=data_path, split="train", download=True, transform=global_transformer())
        test_dataset = multi_mnist_loader.MNIST(root=data_path, split="test", download=True, transform=global_transformer())
    else:
        raise ValueError(f"data set {dataset_name} is not implemented")

    return train_dataset, test_dataset


def global_transformer():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])


import argparse
import json
import os
import logging
from functools import partial
from pathlib import Path
import datetime
import csv

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from network_models import ClassificationBUTDNet, TaskBUTDNet
from utils import *
from names_utils import name2loss, name2metric, name2optim, name2network_module
from mylogger import setup_logger

from datasets import get_dataset, seed_worker


# Output Configurations
WriteCSV = False
SaveLastModel = False
EvalIntermediate = False
EvalInterFrequency = 5
SaveIntermediateModels = False
SavedModelsFrequency = 20

Device = 'cuda'

Settings = "Multi MNIST"


class ModelCTL:
    """
    The main model object for training and evaluating a network.
    """

    def __init__(self, net, **kwargs):
        self.device = Device
        self.net = net.to(self.device)

        self.loss_name = kwargs.get('loss_name', 'BCE')
        self.criterion = name2loss(self.loss_name)
        self.metric_name = kwargs.get('metric_name', 'Accuracy')
        self.metric = name2metric(self.metric_name)

        self.shared_weights = net.shared_weights
        self.task = None

        self.trainloader = None
        self.testloader = None
        self.dataset_name = None

        self.n_classes = net.n_classes

        self.logger = logging.getLogger(__name__)
        self.writer_name = None

    def _analyze(self):
        pass

    def train(self, dataloader=None, lr=0.001, epochs=10, ch_learning=False, **kwargs):
        """
        Train the model's network (self.net)

        Args:
            dataloader (DataLoader): a data loader to train on
            lr (float): the learning rate
            epochs (int): number of epochs to train the model
            ch_learning (bool): whether to use Counter Hebbian Learning or the standard optimizer.
        """
        if dataloader is not None:
            self.trainloader = dataloader

        if self.trainloader is None:
            raise ValueError("Needs to set a train data loader first")

        # learning algorithm parameters
        self.loss_name = kwargs.get('loss_name', self.loss_name)
        self.criterion = name2loss(self.loss_name)
        optimizer = name2optim(kwargs.get('optimizer_name', 'SGD'))(self.net.parameters(), lr=lr)
        if kwargs.get('lr_decay', False):
            if isinstance(kwargs['lr_decay'], bool):
                decay_rate = 0.95
            else:
                decay_rate = kwargs['lr_decay']
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay_rate)
        if ch_learning:
            calc_loss_grads = partial(loss_grads, loss_name=self.loss_name, n_classes=self.n_classes)

        mtl = kwargs.get('mtl', False)
        if mtl:
            if kwargs.get('task', None) is not None:
                self.task = kwargs['task']

        if kwargs.get('writer_name', ''):
            self.writer_name = (datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") + '_' +
                                kwargs.get('writer_name', ''))
        else:
            self.writer_name = (datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") +
                                "_sw" * self.shared_weights +
                                "_chl" * ch_learning +
                                "_mtl" * mtl)

        writer = SummaryWriter(f"runs/{self.writer_name}")

        if EvalIntermediate:
            self._write_metrics(epoch=0, writer=writer, **kwargs)

        self.net.train()
        for epoch in range(epochs):  # loop over the dataset multiple times

            self._analyze()

            running_loss = 0.0

            for i, data in enumerate(self.trainloader):
                # zero the parameter gradients
                optimizer.zero_grad()

                inputs, outputs, labels, tasks = self._inner_op(data, mtl=mtl, train=True)

                # loss = self.criterion(torch.sigmoid(outputs), labels)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                # backward + update
                if ch_learning:
                    # calculate the gradients of the loss with respect to the network outputs
                    # Then propagate it using the TD network and apply the Counter Hebbian learning rule
                    d_l_d_outputs = calc_loss_grads(outputs, labels)
                    self.net.counter_hebbian_back_prop(d_l_d_outputs)
                else:
                    loss.backward()
                optimizer.step()

            if kwargs.get('lr_decay', False):
                lr_scheduler.step()

            epoch_loss = running_loss / len(self.trainloader)
            self.logger.info(f"Epoch {epoch + 1} running loss: {epoch_loss:.3f}")
            writer.add_scalar(f"{self.loss_name} loss/running", epoch_loss, epoch + 1)

            if SaveIntermediateModels and epoch % SavedModelsFrequency == 0:
                self.save_model(self.writer_name + f"epoch_{epoch}.pth")

            if EvalIntermediate and (epoch + 1) % EvalInterFrequency == 0:
                with torch.inference_mode():
                    self._write_metrics(epoch + 1, writer=writer, **kwargs)
                self.net.train()

        self.net.eval()
        self.logger.info('Finished Training')

        if SaveLastModel:
            self.save_model(f"model_{self.dataset_name}" + self.writer_name + '.pth')
        self.logger.info("model is saved")

    def test(self, testloader=None, data_name="Test", **kwargs):
        """
        evaluate the model's network

        Args:
            testloader (DataLoader): the data that will be evaluated. if None, will use self.testloader.
            data_name (str): the name of the data which will be used for the logger.

        Returns:
            scores (dict[str, float]): the scores on the data. keys: loss, metric (for example f1)
        """
        if testloader is not None:
            self.testloader = testloader

        if self.testloader is None:
            raise ValueError("Needs to set a test data loader first")

        self.metric_name = kwargs.get('metric_name', self.metric_name)
        self.metric = name2metric(self.metric_name)

        mtl = kwargs.get('mtl', False)

        self.net.eval()
        running_loss = 0.0

        y_gt = []
        y_preds = []

        for data in self.testloader:
            inputs, outputs, labels, tasks = self._inner_op(data, mtl=mtl, train=False)

            loss = self.criterion(outputs, labels)
            running_loss += loss.item()

            predictions = self.net.predict(outputs)
            y_gt.append(labels.detach().cpu())
            y_preds.append(predictions.detach().cpu())

        total_samples = len(self.testloader)
        final_loss = running_loss / total_samples

        if len(y_gt[0].shape) == 1:
            y_gt = torch.cat([x for x in y_gt])
        else:
            y_gt = torch.argmax(torch.cat([x for x in y_gt]), axis=1)

        metric_score = self.metric(y_gt.numpy(), torch.cat([x for x in y_preds]).numpy())
        self.logger.info(f"{data_name} Loss: {final_loss:.3f}")
        self.logger.info(f"{data_name} {kwargs.get('metric', self.metric_name)}: {metric_score:.3f}")

        results_dict = {f"{self.loss_name} loss": final_loss, kwargs.get('metric', self.metric_name): metric_score}

        return results_dict

    def _inner_op(self, batch_data, mtl=False, train=True):
        inputs = batch_data[0].to(self.device)
        if Settings == "Multi MNIST":
            labels = torch.stack([batch_data[1], batch_data[2]], -1).to(self.device)
        else:
            labels = batch_data[1].to(self.device)

        # forward
        if mtl:
            tasks, labels = self._get_tasks_and_labels(all_labels=labels)
            outputs = self.net.task_guidance_forward(inputs, task=tasks)
        else:
            tasks = None
            outputs = self.net(inputs)

        if len(labels.shape) <= 1 and self.loss_name == 'BCE':
            labels = F.one_hot(labels, self.n_classes).float()

        return inputs, outputs, labels, tasks

    def _get_tasks_and_labels(self, all_labels, tasks=None):
        if self.task == "left/right":
            # if task is not specified, generate a random task
            if tasks is None:
                # Chose randomly whether to the left or right to an object
                tasks = (F.one_hot(torch.randint(0, 2, [all_labels.shape[0]]), 2) * 1.0).to(self.device)

            task_labels = torch.gather(all_labels, 1, tasks.argmax(1).unsqueeze(1))[:, 0]

        else:
            raise ValueError(f"The following task: {self.task} is not supported for multi-task learning")

        return tasks, task_labels

    def save_model(self, model_name):
        """
        save the model's parameters

        Args:
            model_name (str): the model will be saved at `./Saved_Models/model_name`
        """
        Path('Saved_Models').mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), os.path.join('Saved_Models', model_name))

    def load_model(self, model_name):
        """
        load model's parameters from a file, and move them to the device

        Args:
            model_name (str): the model will be loaded from `./Saved_Models/model_name`
        """
        self.net.load_state_dict(torch.load(os.path.join('Saved_Models', model_name)))
        self.net.to(self.device)
        self.net.eval()

    def _write_metrics(self, epoch, writer, **kwargs):
        """
        Write the metrics to tensorboard and optionally to a csv file
        """
        res = self.test(self.trainloader, data_name='Train', **kwargs)
        for k, v in res.items():
            writer.add_scalar(k + '/train', v, epoch)

        if WriteCSV:
            csv_out = [self.dataset_name, self.writer_name, str(epoch)] + [res[k] for k in sorted(res.keys())]

        if self.testloader is not None:
            res = self.test(**kwargs)
            for k, v in res.items():
                writer.add_scalar(k + '/test', v, epoch)

            if WriteCSV:
                csv_out.extend([res[k] for k in sorted(res.keys())])

        if WriteCSV:
            csv_writer.writerow(csv_out)

    def set_dataloaders(self, train_loader, test_loader, dataset_name):
        """
        set the default train data

        Args:
            train_loader (DataLoader): a train data for the model to be trained on
            test_loader (DataLoader): a test data for the model to be evaluated on
            dataset_name (str): the name of the dataset
        """
        self.trainloader = train_loader
        self.testloader = test_loader
        self.dataset_name = dataset_name


def create_data_loaders(data_path=None, dataset_name=None, batch_size=32, **kwargs):
    """
    Create train and test data loaders.

    This function can either load data from an existing directory or generate new data.

    The directory can contain two separate 'train' and 'test' sub-directories.
    Otherwise, split randomly to train and test.

    Args:
        data_path (str): a path to the data set.
            If this path exists, load data from this path. Otherwise, generate a new data to this path.
        data_size (int): the total number of samples which will be generated
        train_percent (float): a percentage of the total data will be used as train.
            The other samples will be used as test.
        batch_size (int): the batch size for the data loaders.

    Returns:
        train_dataloader_ (DataLoader): train data loader
        test_dataloader_ (DataLoader): test data loader
    """
    logger = logging.getLogger(__name__)

    if data_path is None:
        data_path = os.path.join('./Data_Sets', dataset_name)

    datasets_config = {}

    if 'mnist' in dataset_name:
        datasets_config['mnist'] = {
            'path': data_path
        }

    train_dataloader, test_dataloader = get_dataset(dataset_name, batch_size, configs=datasets_config, train=True,
                                                    generator=torch.Generator(), worker_init_fn=seed_worker)

    logger.info("Data Loaders created")
    return train_dataloader, test_dataloader


def update_configs(conf_dict):
    # remove comments
    new_conf_dict = {}
    for k, v in conf_dict.items():
        if isinstance(v, dict):
            new_conf_dict[k] = {}
            for kk, vv in v.items():
                if 'available' not in kk and 'comment' not in kk:
                    new_conf_dict[k][kk] = vv
        else:
            new_conf_dict[k] = v
    del conf_dict

    # automatically infer the data path
    data_set_parameters = new_conf_dict["data set parameters"]
    if data_set_parameters.get('data_path', None) is None:
        data_set_parameters['data_path'] = f"./Data_Sets/{data_set_parameters['dataset_name']}"

    # automatically infer whether the model has shared weights
    if new_conf_dict.get('saved_model_path', None) is not None:
        new_conf_dict['shared_weights'] = 'sw' in new_conf_dict['saved_model_path']

    return new_conf_dict


def plot_parameters(params):
    logger = logging.getLogger(__name__)
    logger.info("The parameters are:")
    for k, v in params.items():
        if isinstance(v, dict):
            logger.info(f"{k}:")
            for kk, vv in v.items():
                logger.info(f"{kk}: {vv}")
        else:
            logger.info(f"{k}: {v}")


def create_network(net_params):
    core_net_arch = name2network_module(net_params['network_name'])

    core_network = core_net_arch(**net_params)

    if net_params.get('mtl', False):
        net = TaskBUTDNet(core_network=core_network, **net_params)
    else:
        net = ClassificationBUTDNet(core_network=core_network, **net_params)

    return net


if WriteCSV:
    date_n_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    csv_f = open(f"runs/{date_n_time_str}.csv", 'w')
    csv_writer = csv.writer(csv_f)
    # TODO: write also headers for the metrics
    csv_writer.writerow(['data set', 'model', 'epoch'])


def main(configs):
    configs = update_configs(configs)
    configs['learning settings'].update(configs['architecture parameters'])

    train_dataloader_, test_dataloader_ = create_data_loaders(**configs['data set parameters'])

    # auto adjust data set input size and number of classes
    update_in_shape_and_n_classes(configs['learning settings'], train_dataloader_)

    plot_parameters(configs)

    net_ = create_network(configs['learning settings'])

    # Define model and data
    model = ModelCTL(net=net_, **configs['evaluation parameters'])
    model.set_dataloaders(train_dataloader_, test_dataloader_, configs['data set parameters']['dataset_name'])

    # train a model or load a pre-trained
    learning_args = {}
    learning_args.update(configs['optimizer parameters'])
    learning_args.update(configs['learning settings'])
    learning_args.update(configs['evaluation parameters'])

    if configs['saved_model_path'] is not None:
        model.load_model(configs['saved_model_path'])
    else:
        model.train(**learning_args)

    with torch.inference_mode():
        model.test(train_dataloader_, 'train', **learning_args)
        model.test(**learning_args)

    if WriteCSV:
        csv_f.close()

    print("finish main")


if __name__ == '__main__':
    now = datetime.datetime.now()
    setup_logger(logging.getLogger(__name__), f"logs_{now.strftime('%Y-%m-%d_%H.%M.%S')}.txt")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=None, help='Path to a config file')
    args = parser.parse_args()

    if args.config_file is not None:
        with open(args.config_file) as config_params:
            configs_ = json.load(config_params)
    else:
        with open("config.json") as config_params:
            configs_ = json.load(config_params)

    main(configs_)

    print('done')

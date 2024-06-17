import argparse
import json
import os
import logging
from functools import partial
from pathlib import Path
import datetime
import csv
import matplotlib
matplotlib.use('TkAgg')

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from butd_modules.butd_data_parallel import MyDataParallel
from data.datasets import get_train_n_test_datasets
from butd_modules.network_models import ClassificationBUTDNet, TaskBUTDNet
from utils import *
from names_utils import name2loss, name2metric, name2optim, name2network_module
from mylogger import setup_logger
from butd_modules.butd_core_networks import BUTDSimpleNet


# Output Configurations
WriteCSV = False
CVSFileName = ''
SaveLastModel = False
SaveIntermediateModels = False
SavedModelsFrequency = 2
SaveForResuming = True
SavedModelDirPath = ''
EvalIntermediate = True
EvalInterFrequency = 2
Analyze = False
AnalyzeFrequency = 2
Repeat = 10

Device = 'cuda'
# Device = 'cpu'

DefaultTestAllTasks = True
DefaultTrainAllTasks = False

WeightDecay = False


class ModelCTL:
    """
    The main model object for training and evaluating a network.
    """

    def __init__(self, net, benchmark='Multi MNIST', **kwargs):
        self.logger = logging.getLogger(__name__)
        self.writer_name = None

        self.benchmark = benchmark

        self.analysis_dict = {}
        
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = Device
        if torch.cuda.device_count() > 1 and self.device != 'cpu':
            self.logger.info(f"{torch.cuda.device_count()} GPUs were detected and will be used")
            # net = torch.nn.DataParallel(net)
            net = MyDataParallel(net)
        self.net = net.to(self.device)

        self.n_classes = net.n_classes
        self.task = kwargs.get('task', None)
        self.n_tasks = kwargs.get('n_tasks', None)

        self.loss_name = kwargs.get('loss_name', 'BCE')
        self.criterion = name2loss(self.loss_name)
        self.metric_name = kwargs.get('metric_name', 'Accuracy')
        self.metric = name2metric(self.metric_name)

        self.trainloader = None
        self.testloader = None
        self.dataset_name = None

        self.csv_writer = kwargs.get('csv_writer', None)

    def _analyze(self, epoch):
        with torch.no_grad():
            if self.benchmark == "Multi MNIST" and isinstance(self.net.core_net, BUTDSimpleNet):
                if 'td_activations' not in self.analysis_dict:
                    self.analysis_dict['td_activations'] = {}

                td_activations = {t: {} for t in range(self.n_tasks)}

                # Sub-Networks Analysis

                # Calculate the task-dependent sub-networks for all tasks
                tasks = torch.eye(self.n_tasks, device=self.device)
                self.net.back_forward(tasks, non_linear=True, lateral=False, task_head=True,
                                      head_non_linear=True, head_lateral=False)

                layer_i = 0
                for td_layer in self.net.core_net.layers:
                    if not hasattr(td_layer, 'td_neurons'):
                        continue
                    for t in range(self.n_tasks):
                        td_activations[t][f"{layer_i} - {td_layer._get_name()}"] = td_layer.td_neurons[t].tolist()
                    layer_i += 1

                self.analysis_dict['td_activations'][epoch] = td_activations
        return None

    def train(self, dataloader=None, lr=0.001, epochs=10, ch_learning=False, train_all_tasks=DefaultTrainAllTasks, **kwargs):
        """
        Train the model's network (self.net)

        Args:
            dataloader (DataLoader): a data loader to train on
            lr (float): the learning rate
            epochs (int): number of epochs to train the model
            ch_learning (bool): whether to use Counter Hebbian Learning or the standard optimizer.
            mtl (bool): whether it is multi-task learning or a single task
            train_all_tasks (bool): in case of multi-task learning, chose whether to sample one random task for each
                instance or to train all the possible tasks.
        """
        if dataloader is not None:
            self.trainloader = dataloader

        if self.trainloader is None:
            raise ValueError("Needs to set a train data loader first")

        # learning algorithm parameters
        self.loss_name = kwargs.get('loss_name', self.loss_name)
        self.criterion = name2loss(self.loss_name)
        if WeightDecay:
            optimizer = name2optim(kwargs.get('optimizer_name', 'SGD'))(self.net.parameters(), lr=lr,
                                                                        weight_decay=kwargs.get('wd_val', WeightDecayValue))
        else:
            optimizer = name2optim(kwargs.get('optimizer_name', 'SGD'))(self.net.parameters(), lr=lr)

        if kwargs.get('lr_decay', False):
            if isinstance(kwargs['lr_decay'], bool):
                decay_rate = 0.95
            else:
                decay_rate = kwargs['lr_decay']
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay_rate)

        calc_loss_grads = partial(loss_grads, loss_name=self.loss_name, n_classes=self.n_classes)

        mtl = kwargs.get('mtl', False)

        if kwargs.get('writer_name', ''):
            self.writer_name = (datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") + '_' +
                                kwargs.get('writer_name', ''))
        else:
            self.writer_name = (datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") +
                                "_sw" * self.net.shared_weights +
                                "_chl" * ch_learning +
                                "_mtl" * mtl +
                                (self.net.multi_decoders * "_multi_d"))

        writer = SummaryWriter(f"runs/{self.writer_name}")
        model_name = (
                f"model_{self.dataset_name}_{self.writer_name}"
        )

        if kwargs.get('resume_training', False):
            # loaded_model_path = os.path.join('Saved_Models', kwargs['saved_model_path'])
            #
            # checkpoint = torch.load(loaded_model_path)
            checkpoint = self.load_model(kwargs['saved_model_path'], get_ckpt_data=True)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if kwargs.get('lr_decay', False):
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            last_epoch = checkpoint['epoch']
            last_loss = checkpoint['loss']

            self.logger.info(f"resuming training, starting from epoch {last_epoch} and loss {last_loss:.3f}")
        else:
            last_epoch = 0

        if EvalIntermediate:
            with torch.inference_mode():
                self._write_metrics(epoch=last_epoch, writer=writer, **kwargs)

        self.net.train()
        if Analyze:
            self._analyze(epoch=0)

        epoch_loss = torch.Tensor([0])
        for epoch in range(last_epoch, epochs):  # loop over the dataset multiple times
            running_loss = 0.0

            for i, data in enumerate(self.trainloader):
                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(not ch_learning):
                    # when learning via ch_learning there is no standard backward
                    if train_all_tasks:
                        tasks_loss = []
                        for task_i in range(self.n_tasks):
                            tasks = (F.one_hot(torch.ones(data[0].shape[0]).long() * task_i, self.n_tasks) * 1.0).to(
                                self.device)
                            inputs, outputs, labels, tasks = self._inner_op(data, mtl=mtl, tasks=tasks, train=True)

                            tasks_loss.append(self.criterion(outputs, labels))

                        loss = torch.mean(torch.stack(tasks_loss))
                    else:
                        inputs, outputs, labels, tasks = self._inner_op(data, mtl=mtl, train=True)

                        loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                # backward + update
                if ch_learning:
                    # calculate the gradients of the loss with respect to the network outputs
                    # Then propagate it using the TD network and apply the Counter Hebbian learning rule
                    d_l_d_outputs = calc_loss_grads(outputs, labels)

                    if kwargs.get('multi_decoders', False):
                        new_dl_do = torch.zeros([d_l_d_outputs.shape[0], 2, d_l_d_outputs.shape[1]], device=d_l_d_outputs.device)
                        new_dl_do[tasks == 1] = d_l_d_outputs
                        d_l_d_outputs = new_dl_do.reshape(d_l_d_outputs.shape[0], -1)

                    self.net.counter_hebbian_back_prop(d_l_d_outputs)
                else:
                    loss.backward()
                optimizer.step()

                if (i+1) % 100 == 0:
                    self.logger.info(f"iteration {i + 1} running loss: {running_loss / (i+1):.3f}")

            if Analyze and (epoch + 1) % AnalyzeFrequency == 0:
                self._analyze(epoch=epoch + 1)

            if kwargs.get('lr_decay', False):
                lr_scheduler.step()

            epoch_loss = running_loss / len(self.trainloader)
            self.logger.info(f"Epoch {epoch + 1} running loss: {epoch_loss:.3f}")
            writer.add_scalar(f"{self.loss_name} loss/running", epoch_loss, epoch + 1)

            if SaveIntermediateModels and epoch % SavedModelsFrequency == 0:
                if SaveForResuming:
                    checkpoint_dict = {
                        'epoch': epoch + 1,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss,
                    }
                    if kwargs.get('lr_decay', False):
                        checkpoint_dict['lr_scheduler_state_dict'] = lr_scheduler.state_dict()

                    self.save_model(f"{model_name}_epoch_{epoch+1}.tar", checkpoint_dict=checkpoint_dict)

                else:
                    self.save_model(f"{model_name}_epoch_{epoch+1}.pth")

            if EvalIntermediate and (epoch + 1) % EvalInterFrequency == 0 and epoch + 1 != epochs:
                with torch.inference_mode():
                    self._write_metrics(epoch + 1, writer=writer, **kwargs)
                self.net.train()

        self.net.eval()
        self.logger.info('Finished Training')

        # Evaluate the learned model
        with torch.inference_mode():
            self._write_metrics(epoch=epochs, writer=writer, **kwargs)

        if SaveLastModel:
            self.save_model(f"{model_name}.pth")
            if SaveForResuming:
                checkpoint_dict = {
                    'epoch': last_epoch + epochs,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                }
                if kwargs.get('lr_decay', False):
                    checkpoint_dict['lr_scheduler_state_dict'] = lr_scheduler.state_dict()

                self.save_model(f"{model_name}tar", checkpoint_dict=checkpoint_dict)
        self.logger.info("model is saved")

    def test(self, testloader=None, data_name="Test", test_all_tasks=DefaultTestAllTasks, **kwargs):
        """
        evaluate the model's network

        Args:
            testloader (DataLoader): the data that will be evaluated. if None, will use self.testloader.
            data_name (str): the name of the data which will be used for the logger.

        Returns:
            scores (dict[str, float]): the scores on the data. keys: loss, metric (for example f1)
        """
        if testloader is None and self.testloader is not None:
            testloader = self.testloader

        if testloader is None:
            raise ValueError("Needs to set a test data loader first")

        self.metric_name = kwargs.get('metric_name', self.metric_name)
        self.metric = name2metric(self.metric_name)

        mtl = kwargs.get('mtl', False)

        self.net.eval()
        running_loss = 0.0

        y_gt = []
        y_preds = []

        if test_all_tasks:
            for data in testloader:
                for task_i in range(self.n_tasks):
                    tasks = (F.one_hot(torch.ones(data[0].shape[0]).long()*task_i, self.n_tasks) * 1.0).to(self.device)
                    inputs, outputs, labels, tasks = self._inner_op(data, mtl=mtl, tasks=tasks, train=False)

                    loss = self.criterion(outputs, labels)
                    running_loss += loss.item()

                    predictions = self.net.predict(outputs)
                    y_gt.append(labels.detach().cpu())
                    y_preds.append(predictions.detach().cpu())
        else:
            for data in testloader:
                inputs, outputs, labels, tasks = self._inner_op(data, mtl=mtl, train=False)

                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                predictions = self.net.predict(outputs)
                y_gt.append(labels.detach().cpu())
                y_preds.append(predictions.detach().cpu())

        if test_all_tasks:
            total_samples = len(testloader) * self.n_tasks
        else:
            total_samples = len(testloader)

        final_loss = running_loss / total_samples

        if len(y_gt[0].shape) == 1:
            y_gt = torch.cat([x for x in y_gt])
        elif len(y_gt[0].shape) == 2 and y_gt[0].shape[1] == 1:
            y_gt = torch.cat([x[:, 0] for x in y_gt])
        else:
            y_gt = torch.argmax(torch.cat([x for x in y_gt]), axis=1)

        y_preds = torch.cat([x for x in y_preds])

        metric_score = self.metric(y_gt.numpy(), y_preds.numpy())
        self.logger.info(f"{data_name} Loss: {final_loss:.4f}")
        self.logger.info(f"{data_name} {kwargs.get('metric', self.metric_name)}: {metric_score:.5f}")

        results_dict = {f"{self.loss_name} loss": final_loss, kwargs.get('metric', self.metric_name): metric_score}

        return results_dict

    def _inner_op(self, batch_data, mtl=False, tasks=None, train=True):
        inputs = batch_data[0].to(self.device)
        if self.benchmark == "Multi MNIST":
            labels = torch.stack([batch_data[1], batch_data[2]], -1).to(self.device)
        else:
            labels = batch_data[1].to(self.device)

        # forward
        if mtl:
            tasks, labels = self._get_tasks_and_labels(all_labels=labels, tasks=tasks)
            outputs = self.net(inputs, task=tasks)
        else:
            tasks = None
            outputs = self.net(inputs)

        if self.loss_name in ['BCE', 'MSE']:
            if len(labels.shape) <= 1 and (not self.net.multi_decoders):
                if self.n_classes == 1:
                    labels = labels.unsqueeze(-1)
                else:
                    labels = F.one_hot(labels, self.n_classes)
            labels = labels.float()

        return inputs, outputs, labels, tasks

    def _get_tasks_and_labels(self, all_labels, tasks=None):
        if self.task == "left/right of":
            # if task is not specified, generate a random task
            if tasks is None:
                # Chose randomly whether to the left or right to an object
                tasks = (F.one_hot(torch.randint(0, 2, [all_labels.shape[0]]), 2) * 1.0).to(self.device)

                # Given all labels that appear in the input ordered according to their position
                # Extract all the indices of all labels that appear in the input
                if len(all_labels.shape) > 2:
                    # If all_labels is a one-hot vector, get the class indices
                    all_labels = torch.argmax(all_labels, -1)
                # Pick a random label.
                # If the task is to predict the object to the left, don't pick the first,
                # If it is to the right, don't pick the last
                chosen_locations = (torch.randint(
                    0, all_labels.shape[-1] - 1,
                    [all_labels.shape[0]]).to(self.device) + tasks[:, 0].type(torch.int64)
                                    ).unsqueeze(1)
                chosen_labels = torch.gather(all_labels, 1, chosen_locations)[:, 0]

                # Concatenate the chosen label to the task
                tasks = torch.cat([tasks, F.one_hot(chosen_labels, self.n_classes)], -1)

            # calculate the ground truth labels of that task
            # Find the instance requested in the task
            instance_location = (tasks[:, 2:].argmax(1).unsqueeze(1) == all_labels).nonzero()[:, 1]

            # Find the target instance location - to the left/right of the instance mentioned in the task
            target_instance_location = instance_location - tasks[:, 0] + tasks[:, 1]

            # Get the label at that location
            task_labels = torch.gather(all_labels, 1, target_instance_location.type(torch.int64).unsqueeze(1))[:, 0]
        elif self.task == "left/right":
            # if task is not specified, generate a random task
            if tasks is None:
                # Chose randomly whether to the left or right to an object
                tasks = (F.one_hot(torch.randint(0, 2, [all_labels.shape[0]]), 2) * 1.0).to(self.device)

            task_labels = torch.gather(all_labels, 1, tasks.argmax(1).unsqueeze(1))[:, 0]
        elif self.task == "binary attribute":
            if tasks is None:
                # Chose randomly whether to the left or right to an object
                tasks = (F.one_hot(torch.randint(0, self.n_tasks, [all_labels.shape[0]]), self.n_tasks) * 1.0).to(self.device)

            task_labels = torch.gather(all_labels, 1, tasks.argmax(1).unsqueeze(1))[:, 0]
        else:
            raise ValueError(f"The following task: {self.task} is not supported for multi-task learning")

        return tasks, task_labels

    def save_model(self, model_name, checkpoint_dict=None):
        """
        save the model's parameters

        Args:
            model_name (str): the model will be saved at `./Saved_Models/model_name`
            checkpoint_dict (dict): a dictionary contain relevant information for resuming training.
        """
        if os.path.exists(SavedModelDirPath) and SavedModelDirPath != '':
            dir_path = SavedModelDirPath
        else:
            Path('Saved_Models').mkdir(parents=True, exist_ok=True)
            dir_path = 'Saved_Models'

        if checkpoint_dict is None:
            torch.save(self.net.state_dict(), os.path.join(dir_path, model_name))
        else:
            checkpoint_dict.update({'model_state_dict': self.net.state_dict()})

            torch.save(checkpoint_dict, os.path.join(dir_path, model_name))

    def load_model(self, model_name, get_ckpt_data=False):
        """
        load model's parameters from a file, and move them to the device

        Args:
            model_name (str): the model will be loaded from `./Saved_Models/model_name`
        """
        if os.path.exists(SavedModelDirPath) and SavedModelDirPath != '':
            dir_path = SavedModelDirPath
        else:
            Path('Saved_Models').mkdir(parents=True, exist_ok=True)
            dir_path = 'Saved_Models'

        model_path = os.path.join(dir_path, model_name)

        if not os.path.exists(model_path):
            raise ValueError(f"try to load {model_name}, but it was not found")

        loaded_data = torch.load(model_path)

        if '.tar' in model_name:
            self.net.load_state_dict(loaded_data['model_state_dict'])
        else:
            self.net.load_state_dict(loaded_data)

        self.net.to(self.device)
        self.net.eval()

        if get_ckpt_data:
            del loaded_data['model_state_dict']
            return loaded_data

    def _write_metrics(self, epoch, writer, **kwargs):
        """
        write the metrics for tensorboard
        """
        self.logger.info(f"Evaluating epoch {epoch}:")
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
            self.csv_writer.writerow(csv_out)

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


def create_data_loaders(data_path=None, dataset_name=None, batch_size=64, **kwargs):
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

    train_dataset, test_dataset = get_train_n_test_datasets(dataset_name, data_path, **kwargs)

    # Create Data Loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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


def main(configs):
    configs = update_configs(configs)
    configs['learning settings'].update(configs['architecture parameters'])

    train_dataloader_, test_dataloader_ = create_data_loaders(**configs['data set parameters'])

    # auto adjust data set input size and number of classes
    update_in_shape_and_n_classes(configs['learning settings'], train_dataloader_)

    plot_parameters(configs)

    if WriteCSV:
        Path("runs").mkdir(parents=False, exist_ok=True)
        date_n_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        csv_f = open(f"runs/{date_n_time_str}_{CVSFileName}.csv", 'w')
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow(['data set', 'model', 'epoch', 'train acc', 'train loss', 'test acc', 'test loss'])
    else:
        csv_writer = None

    for _ in range(Repeat):

        net_ = create_network(configs['learning settings'])

        # Define model and data
        model = ModelCTL(net=net_, benchmark=configs.get('benchmark', 'Multi MNIST'), **configs['learning settings'],
                         csv_writer=csv_writer)
        model.set_dataloaders(train_dataloader_, test_dataloader_, configs['data set parameters']['dataset_name'])

        learning_args = {}
        learning_args.update(configs['optimizer parameters'])
        learning_args.update(configs['learning settings'])
        learning_args['saved_model_path'] = configs['saved_model_path']

        if learning_args['saved_model_path'] is not None and not learning_args['resume_training']:
            # load a pre-trained
            model.load_model(learning_args['saved_model_path'])
        else:
            if learning_args['resume_training'] and learning_args['saved_model_path'] is None:
                raise ValueError("'saved_model_pah' must be provided when resuming training")
            if learning_args['resume_training'] and '.tar' not in learning_args['saved_model_path']:
                raise ValueError(f"{learning_args['saved_model_pah']} is not a resumeable file, should be a '.tar' file")
            model.train(**learning_args)

        with torch.inference_mode():
            # test_res = model.test(train_dataloader_, 'train', test_all_tasks=True, **learning_args)
            model.test(test_all_tasks=True, **learning_args)

    if WriteCSV:
        csv_f.close()

    print("finish main")


if __name__ == '__main__':
    now = datetime.datetime.now()
    setup_logger(logging.getLogger(__name__), f"logs_{now.strftime('%Y-%m-%d_%H.%M.%S')}.txt")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='config.json', help='Path to a config file')
    args = parser.parse_args()

    if args.config_file is not None:
        with open(args.config_file) as config_params:
            configs_ = json.load(config_params)
    else:
        raise ValueError(f'The config file {args.config_file} does not exist')

    main(configs_)

    print('done')

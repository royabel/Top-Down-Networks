
# Top-Down Network Combines Back-Propagation with Attention

This repository is implementation of the paper: "Top-Down Processing: Top-Down Network Combines Back-Propagation with Attention". 

## The Method

We use a bottom-up (BU) - top-down (TD) model, in which a single TD network is used for both learning and guiding TD attention.

This model enables: 
*   Multi-task learning (MTL), by dynamically learning task-dependent sub-networks for each task (see Section 5 in the paper). 
*   Counter-Hebbian learning, a biologically motivated learning algorithm (see Section 4 in the paper).

See the paper for more details.


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Multi-Task Learning 

To reproduce the Multi-MNIST experiments in the paper, run this command:

```train
python main.py --config_file multi_mnist_config.json 
```

** Note that the Multi-MNIST dataset will be automatically downloaded to your directory

To reproduce the Multi-MNIST experiments in the paper, run this command:

```train
python main.py --config_file celeb_a_config.json 
```

** Note that the dataset must be downloaded in advance either from [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) or from Pytorch torchvision datasets

## BU-TD Modules

BU-TD layers wrap standard Pytorch layers to fit a BU-TD model. 
*   Implements a `back_forward` method. The `back_forward` method serves as a forward pass for the TD network, while the `forward` method is for the BU.
*   It allows lateral connectivity, each network can gate the other. 

Some layers already implemented in `butd_layer.py` (e.g. Linear, Conv2d), you can add your own layers. 

ResNet blocks are implemented in `butd_building_blocks.py`.  

## Creating a BU-TD model

*   Define your own BU-TD model using the custom layers.
    *   Use BU-TD layers similar to the standard PyTorch layers usage. 
    *   The BU-TD model class must include `back_forward` method (and optionally `counter_hebbian_update_value` method to support Counter-Hebbian learning)
    *   ResNet and Convolutional networks examples can be found in `butd_core_networks.py`.
    
## Counter-Hebbian Learning

To learn via Counter-Hebbian learning, each PyTorch Module must contain a `counter_hebbian_update_value` method. 
This method fill the gradients attribute of the weights with the update values derived by the Counter-Hebb learning rule. 
These value can be used in standard PyTorch optimizers. 
Therefore, do not use `loss.backward` together with Counter-Hebbian learning.   

## Running an Experiment

To run an experiment, set hyper-parameters and other configurations in a `config.json` file, and run this command:  

```run
python main.py --config_file config.json
```

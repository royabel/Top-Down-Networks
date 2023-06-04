
# Biologically Inspired Learning Model for Instructed Vision

This repository is the official implementation of the paper: ["Biologically Inspired Learning Model for Instructed Vision"](https://arxiv.org/abs/2306.02415),
presented on *NeurIPS 2024*. 

Visit the conference page for additional resources: [link](https://neurips.cc/virtual/2024/poster/94152)  

The paper propose a biologically inspired learning method for instruction-models. 


The key contributions of the paper are:

* Propose a biologically inspired learning model for instructed models. 

* Propose a novel top-down mechanism that combines learning from error signals with guiding top-down attention.

* Suggest a Counter-Hebb-based learning procedure, that can perform the exact backpropagation.

* Extending earlier work, offering a new step toward a more biologically plausible learning model.


The key features of this library are:
* Transforms conventional networks into instruction-based models, allowing task-dependent sub-networks to be selected without adding additional parameters.
* Implements Counter-Hebb learning, enabling training with either standard backpropagation or the Counter-Hebb learning rule.

## The Method

Our model integrates two networks: a bottom-up (BU) network and a top-down (TD) network. 
A single TD network is used for both learning and guiding attention.
For parameter efficiency, the two networks can share the same set of weights. 

## Learning Instructed Models 
In instructed models, the network takes an input signal and an instruction, then predicts a response based on both. For example, in instructed vision tasks, the model could answer a question about an image.

The inference phase consists of two passes:

1. Top-Down (TD) Pass: Processes the instruction, selecting a task-dependent sub-network within the full network.
2. Bottom-Up (BU) Pass: Processes the input signal (e.g., image) through the selected sub-network to generate a prediction.


Two training options are supported for the model

* Standard backpropagation
* Biologically-Inspired Learning, where the TD network propagates errors, followed by a Counter-Hebb update to adjust the weights of both the BU and TD networks.


Below, we illustrate the biologically inspired learning algorithm, detailing the three passes of the networks:


<img src="/figs/learning_instructed_vision_schematic.png" style="width:80%; height:auto;">

[//]: # (![MTL]&#40;/figs/learning_instructed_vision_schematic.png&#41;)



## Guided Processing 
Zooming in on the BU Step
In this step, instructions guide image processing by selecting specific sub-networks within the bottom-up network. The image is processed exclusively on the selected sub-network.
The figure illustrates this modular architecture, where each instruction activates a distinct sub-network (highlighted in dark).

<img src="/figs/guided_bu.png" style="width:40%; height:auto;">

[//]: # (![Guided Processing]&#40;/figs/guided_bu.png&#41;)

### Counter-Hebbian Learning
A biologically inspired learning mechanism. Similar to classical Hebbian learning, the Counter-Hebb rule updates synapses (weights) based on the activity of connected neurons. 
See figure below for an illustration. For more details, refer to the paper.


<img src="/figs/Counter-Hebb.png" style="width:40%; height:auto;">

[//]: # (![CH learning]&#40;/figs/Counter-Hebb.png&#41;)



## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Experiments

To reproduce the experiments in the paper, run the following command with the proper config file:

```train
python main.py --config_file <name of a config file> 
```

The following config files correspond to the different experiments in the paper:
* `celeb_a_config.json`
* `multi_mnist_config.json`
* `cifar_config.json`
* `fashion_mnist_config.json`
* `mnist_config.json`

You can modify the experiments by modifying the config files 

Note that all data set but the Celeb-A dataset will be automatically downloaded to your directory. 
For the Celeb-A experiments you must downloaded the dataset in advance either from [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) or from Pytorch torchvision datasets

## Running an Experiment

To run a custom experiment, set hyper-parameters and other configurations in a `config.json` file, and run this command:  

```run
python main.py
```

## Implementation Details
We detail the implementation of BU-TD models using bottom-up (BU) and top-down (TD) networks.  

### BU-TD Modules

BU-TD layers wrap standard Pytorch layers to fit a BU-TD model. 
*   Implements a `back_forward` method. The `back_forward` method serves as a forward pass for the TD network, while the `forward` method is for the BU.
*   It allows lateral connectivity, each network can gate the other. 

Some layers already implemented in `butd_layer.py` (e.g. Linear, Conv2d), you can add your own layers. 

ResNet blocks are implemented in `butd_building_blocks.py`.  

### Creating a BU-TD model

Define your own BU-TD model using the custom BU-TD modules as layers.
*   Use BU-TD layers similar to the standard PyTorch layers usage. 
*   The BU-TD model class must include `back_forward` method (and optionally `counter_hebbian_update_value` method to support Counter-Hebbian learning)
*   ResNet and other examples can be found in `butd_architectures.py`.
    
### Counter-Hebbian Learning

To enable Counter-Hebb learning, each PyTorch module must implement a `counter_hebbian_update_value` method. 
This method fills the gradient attribute of the weights with update values computed using the Counter-Hebb learning rule.
These values can then be utilized with standard PyTorch optimizers.
Note: Do not use `loss.backward` in conjunction with Counter-Hebb learning.


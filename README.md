
# Top-Down Network Combines Back-Propagation with Attention

This repository is the implementation of the paper: ["Biologically-Motivated Learning Model for Instructed Visual Processing"](https://arxiv.org/abs/2306.02415).

The paper propose a biologically-inspired learning method for instruction-models. It uses a bottom-up (BU) - top-down (TD) model, in which a single TD network is used for both learning and guiding attention.

[//]: # (The key contributions of the paper are:)

[//]: # (* Propose a biologically motivated learning model for instructed models. )

[//]: # (* Propose a novel top-down mechanism that combines learning from error signals with guiding top-down attention.)

[//]: # (* Extending earlier work, offering a new step toward a more biologically plausible learning model. )

[//]: # (* suggest a Counter-Hebb learning procedure &#40;synaptic modification&#41; that can perform the exact backpropagation. )

[//]: # (* Present a novel biologically-inspired MTL algorithm that dynamically creates unique task-dependent sub-networks within conventional networks. )

The key features of this library are:
* Converts conventional networks into an instruction based model (without adding additional parameters). 
* Given an instruction/task, select a task-dependent sub-network within the full network to perform the task. 
* The models can be learned either by the standard backpropagation or by 'Counter-Hebb' learning

## The Method

![BU-TD Approach](/figs/top_down_processing.png)

The proposed BU-TD approach consists of BU (blue $\uparrow$) and TD (orange $\downarrow$) networks with bi-directional connections. These networks can operate recurrently. A single TD network is used for both propagating down error signals and TD attention, while the BU network handles the processing of input signals. On the right, this concept is illustrated within a multi-task learning setting. 
The input for each component is indicated by a letter: $I$ marks the input signal (e.g. images in the case of vision), $E$ marks error signals (e.g. loss gradients), and $A$ marks attention signals, e.g. selected object, location, or task. 


### Counter-Hebbian Learning
A biologically motivated learning mechanism. Similar to the classical Hebbian learning, the Counter-Hebb learning rule update the synapse based on the activity of the neurons connected to the synapse (please see the paper for more information).

![CH learning](/figs/update_rule.png)

### Guided Learning (multi-task)
The Multi-Task Learning (MTL) algorithm comprises of two phases: a TD pass followed by a BU pass for prediction, and another TD pass for 'Counter-Hebb' learning. The last TD pass can be replaced by backpropagation.  
**Inference:**
The selected task provides input to the TD network which select a task-dependent sub network within the full network. The BU network then processes the input signals (image) using only the selected sub-network to generate prediction.
**Learning:**
This model can be learned via the standard backpropagation.
Alternatively, the same TD network can be reused to propagate prediction error signals that used in 'Counter-Hebb' learning.

![MTL](/figs/MTL_schematic.png)

See the paper for more details.


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
* 'celeb_a_config.json'
* 'multi_mnist_config.json'
* 'cifar_config.json'
* 'fashion_mnist_config.json'
* 'mnist_config.json'

You can modify the experiments by modifying the config files 

Note that all data set but the Celeb-A dataset will be automatically downloaded to your directory. 
For the Celeb-A experiments you must downloaded the dataset in advance either from [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) or from Pytorch torchvision datasets

## BU-TD Modules

BU-TD layers wrap standard Pytorch layers to fit a BU-TD model. 
*   Implements a `back_forward` method. The `back_forward` method serves as a forward pass for the TD network, while the `forward` method is for the BU.
*   It allows lateral connectivity, each network can gate the other. 

Some layers already implemented in `butd_layer.py` (e.g. Linear, Conv2d), you can add your own layers. 

ResNet blocks are implemented in `butd_building_blocks.py`.  

## Creating a BU-TD model

Define your own BU-TD model using the custom layers.
*   Use BU-TD layers similar to the standard PyTorch layers usage. 
*   The BU-TD model class must include `back_forward` method (and optionally `counter_hebbian_update_value` method to support Counter-Hebbian learning)
*   ResNet and other examples can be found in `butd_architectures.py`.
    
## Counter-Hebbian Learning

To learn via Counter-Hebbian learning, each PyTorch Module must contain a `counter_hebbian_update_value` method. 
This method fill the gradients attribute of the weights with the update values derived by the Counter-Hebb learning rule. 
These value can be used in standard PyTorch optimizers. 
Therefore, do not use `loss.backward` together with Counter-Hebbian learning.   

## Running an Experiment

To run a custom experiment, set hyper-parameters and other configurations in a `config.json` file, and run this command:  

```run
python main.py
```

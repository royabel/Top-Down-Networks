
# Top-Down Network Combines Back-Propagation with Attention

This repository is the implementation of the paper: ["Top-Down Network Combines Back-Propagation with Attention"](https://arxiv.org/abs/2306.02415).

The paper propose a biologically-inspired learning method for instruction-models. It uses a bottom-up (BU) - top-down (TD) model, in which a single TD network is used for both learning and guiding attention.
The key contributions of the paper are:
* Propose a novel top-down mechanism that combines learning from error signals with top-down attention.
* Extending earlier work, offering a new step toward a more biologically plausible learning model. 
* Present a novel biologically-inspired MTL algorithm that dynamically creates unique task-dependent sub-networks within conventional networks. 

## The Method

![BU-TD Approach](/figs/top_down_processing.png)

The proposed BU-TD approach consists of BU (blue $\uparrow$) and TD (orange $\downarrow$) networks with bi-directional connections. These networks can operate recurrently. A single TD network is used for both propagating down error signals and TD attention, while the BU network handles the processing of input signals. On the right, this concept is illustrated within a multi-task learning setting. 
The input for each component is indicated by a letter: $I$ marks the input signal (e.g. images in the case of vision), $E$ marks error signals (e.g. loss gradients), and $A$ marks attention signals, e.g. selected object, location, or task. 


This model enables: 

### Counter-Hebbian Learning
A biologically motivated learning algorithm. The Counter-Hebb update rule in comparison with the classical Hebb rule. Focusing on a single upstream synapse (outlined by a circle), connecting a pre-synaptic neuron with a post-synaptic neuron. Both rules update the synapse based on the activity of both associated neurons. However, the Counter-Hebb update rule, presented on the right, relies on the counterpart downstream (marked in orange) counter neurons which is connected via lateral connections instead of a back firing from the upstream. 
![CH learning](/figs/update_rule.png)

### Multi-task Learning (MTL)
The MTL algorithm offers dynamically learning task-dependent sub-networks for each task.
The MTL algorithm comprises of two phases: a TD pass followed by a BU pass for prediction, and another TD pass for learning. The selected task provides input to the TD network via the task head $\Bar{H}_{\text{task}}$, and the activation propagates downward attention-guiding signals with ReLU non-linearity. By applying ReLU, the task selectively activates a subset of neurons (i.e. non-zero values), composing a sub-network within the full network. The BU network then processes an input image using a composition of ReLU and GaLU. The GaLU function (denoted with dashed arrows) gates the BU hidden layers $h_i$ by their corresponding counter TD hidden layers $\bar{h}_i$. As a result, the BU computation is performed only on the selected sub-network. Lastly, the prediction head $H_{\text{pred}}$ generates a prediction based on the top-level BU hidden layer $h_L$. For learning, the same TD network is then reused to propagate prediction error signals, starting from the prediction head $\Bar{H}_{\text{pred}}$. This computation is performed with GaLU exclusively (no ReLU), thereby permitting negative values. Finally, the 'Counter-Hebb' learning rule adjusts both networks' weights based on the activation values of their hidden layers $h_i$ and $\bar{h}_i$. Therefore, in contrast with standard models, the entire computation is carried out by neurons in the network, and no external computation is used for learning (e.g. Back-Propagation).
Alternatively, the second phase can be replabed with standard BP under the constraints of sharing the BU and TD weights. This yields an equivalent learning phase.  
![MTL](/figs/MTL_schematic.png)


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

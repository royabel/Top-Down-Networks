
# Top-Down Processing: Top-Down Network Combines Back-Propagation with Attention

This repository is the official implementation of [Top-Down Processing: Top-Down Network Combines Back-Propagation with Attention](https://arxiv.org/abs/2306.02415). 

## The Method

We use a bottom-up (BU) - top-down (TD) model, in which a single TD network is used for both learning and guiding TD attention.

This model enables: 
*   Multi-task learning (MTL), by dynamically learning task-dependent sub-networks for each task. 
*   Counter-Hebbian learning, a biologically motivated learning algorithm.

![Top-Down processing](/figs/td_processing.png)

The different approaches of combining BU with TD processing in deep learning models. Solid arrows represent computations that are part of the model, dashed arrows represent an external backward computation that is not part of the model. (a) illustrates standard bottom-up architectures followed by an external back-propagation. (b) illustrates current methods for using TD attention. A TD stream creates an attention signal that influences the subsequent BU processing, followed by external backward computation for updating both BU and TD weights. Other architectures that share the same concept have been proposed, for instance, a popular scheme termed ‘U-net’ is similar to (b), but applies a BU stage first, followed by TD. (c) Our BU-TD model consists of BU and TD networks with bi-directional connections. A single TD network is used for both back-propagating errors and TD attention. 

For example, in multi-task learning (MTL) on Multi-MNIST, the selected task (left/right) provides input to the TD network which propagates downwards. This TD pass guides the next BU pass by selecting a sub-network within the full network. The BU network then receives an input image and generates a prediction by performing only on the partial sub-network. The same TD network then can be used to propagate error signals for updating the weights without any additional backward computation.

![Top-Down processing](/figs/MTL.png)


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Multi-Task Learning 

To reproduce the experiment in the paper, run this command:

```train
python main.py --config_file multi_mnist_config.json 
```

Note that the Multi-MNIST dataset will be automatically downloaded to your directory

## BU-TD layers

BU-TD layers wrap standard Pytorch layers to fit a BU-TD model. 
*   Implements a `back_forward` method. The `back_forward` method serves as a forward pass for the TD network, while the `forward` method is for the BU.
*   It allows lateral connectivity, each network can gate the other. 

Many layers already implemented in `custom_layer.py`, you can add your own layers. 

ResNet blocks are implemented in `butd_network_building_blocks.py`.  

## Creating a BU-TD model

*   Define your own BU-TD model using the custom layers.
    *   Use BU-TD layers similar to the standard PyTorch layers usage. 
    *   The BU-TD model class must include `back_forward` method (and optionally `counter_hebbian_update_value` method to support Counter-Hebbian learning)
    *   ResNet and Convolutional networks examples can be found in `core_networks.py`.
    
## Counter-Hebbian Learning

To learn via Counter-Hebbian learning, each PyTorch Module must contain a `counter_hebbian_update_value` method. 
This method fill the gradients attribute of the weights with the update values derived by the Counter-Hebb learning rule. 
These value can be used in standard PyTorch optimizers. 
Therefore, do not use `loss.backward` together with Counter-Hebbian learning.   

## Running an Experiment

To run an experiment, set hyper-parameters and other configurations in the `config.json` file, and run this command:  

```run
python main.py 
```

# [FInCFlow: Fast and Invertible k×k Convolutions for Normalizing Flows](https://arxiv.org/abs/2301.09266)
## [Paper](https://www.scitepress.org/Link.aspx?doi=10.5220/0011876600003417) accepted: [VISAPP'23](https://visapp.scitevents.org/Home.aspx)

## Thesis Defence Presentation: ![Presentation](misc/Thesis_Defence.pdf) 

![Flow overview](misc/FInCFlow_model.png)
Generating Images using Fast Normalizing Flows with Invertible Convolution

![Flow overview](misc/inverse_direct2.gif)
Inverse calculation of a Lower Triangular matrix with Substitution Method
![Flow overview](misc/inverse_finc2.gif)
Inverse calculation of a Lower Triangular MAtrix from our method

## Developing in ada
- create a python3.9 venv with all packages in requirments.txt installed in ~/venv/fastflow
- source env.sh
## requirements 
`pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu102`
## Experiments

###Environment Setup
1. Make sure you have Anaconda or Miniconda installed.
2. Clone repo with `git clone https://github.com/aditya-v-kallappa/FInCFlow/.`
3. Go into the cloned repo: `cd FastFlow`
4. Create the environment: `conda env create ff`
5. Activate the environment: `source activate ff`

## Getting Started
#### Install requirements with Anaconda:
`conda env create -f conda_environment.yml`

#### Install snf package
Install the snf package locally for development. This allows you to run experiments with the `fastflow` command. At the root of the project directory run:
`pip install -e .`

Note that this package requires Ninja to build the C++ extensions required to efficiently compute the Fast-Flow gradient, however this should be installed automatically with pytorch or 

`pip install Ninja`

#### (Optional) Setup Weights & Biases:
This repository uses Weight & Biases for experiment tracking. By deafult this is set to off. However, if you would like to use this (highly recommended!) functionality, all you have to do is install weight & biases as follows, and then set `'wandb': True`,  `'wandb_project': YOUR_PROJECT_NAME`, and `'wandb_entity': YOUR_ENTITY_NAME` in the default experiment config at the top of snf/train/experiment.py.

To install Weights & Biases follow the [quickstart guide here](https://docs.wandb.com/quickstart), or, simply run `pip install wandb` (with the pip associated to your conda environment), followed by: `wandb login`

Make sure that you have first [created a weights and biases account](https://app.wandb.ai/login?signup=true), and filled in the correct `wandb_project` and `wandb_entity` in the experiment config.


## Basics of the framework
- All models are built using the `FlowSequential` module (see fastflow/layers/flowsequential.py)
    - This module iterates through a list of `FlowLayer` or `ModifiedGradFlowLayer` modules, repeatedly transforming the input, while simultaneously accumulating the log-determinant of the jacobian of each transformation along the way.
    - Ultimately, this layer returns the total normalized log-probability of input by summing the log-probability of the transformed input under the base distribuiton, and the accumulated sum of log jacobian determinants (i.e. using the change of variables rule).
- The `Experiment` class (see fastflow/train/experiment.py) handles running the training iterations, evaluation, sampling, and model saving & loading.
- All experiments can be found in `fastflow/experiments/`, and require the specification of a model, optimizer, dataset, and config dictionary. See below for the currently implemented options for the config.  
- All layer implementations can be found in `fastflow/layers/` including the fastflow layer found at fastflow/layers/fastflow.py

## Overview of Config options
The configuration dictionary is mainly used to modify the training procedure specified in the Experiment class, but it can also be used to modify the model architecture if desired. A non-exhaustive list of config options and descriptions are given below, note that config options which modify model architecture may not have any effect if they are not explicitly incorporated in the `create_model()` function of the experiments. The default configuration options are specified at the top of the experiment file. The configuration dictionary passed to the Experiment class initalizer then overwrites these defaults if the key is present.


#### Training Options (important first)
- `'modified_grad'`: *bool*, if True, use the fastflow gradient in place of the true exact gradient. This also causes fastflow layers to return 0 for their log-jacobian-determinant during training. During evaluation, this option has no effect, and log-likelihoods are computed exactly.  

- `'lr'`: *float*, learning rate
- `'epochs'`: *int*, total training epochs
- `'eval_epochs'`: *int*, number of epochs between computing true log-likelihood on validation set.
- `'eval_train'`:  *bool*, if True, also compute true-log-likelihood on train set during eval phase.
- `'max_eval_ex'`: *int*, maximum number of examples to run evaluation on (default `inf`). Useful for models with extremely computationally expensive inference procedures. 
- `'warmup_epochs'`: *int*, number of epochs over which to linearly increase learning rate from $0$ to `lr`.
- `'batch_size'`: *int*, number of samples per batch
- `'recon_loss_weight'`: *float*, Value of $\lambda$ weight on reconstruction gradient. 
- `'sym_recon_grad'`: Bool, if True, and `add_recon_grad` is true, use a symmetric version of the reconstruction loss for increased stability.
- `'grad_clip_norm'`: *float*, maximum magnitude which to scale gradients to if greater than.

#### Model Architecutre Options
- `'activation'`: *str*: Name of activation function to use for FC and CNN models (one of `SLR, LLR, Spline, SELU`).
- `'num_blocks'`: *int*, Number of blocks for glow-like models
- `'block_size'`: *int*, Number of setps-of-flow per block in glow-like models
- `'num_layers'`: *int*, Number of layers for CNN and FC models
- `'actnorm'`: *bool*, if True, use ActNorm in glow-like models
- `'split_prior'`: *bool*, if True, use Split Priors between each block in glow-like models

#### Logging Options
- `'wandb'`: *bool*, if True, use weights & biases logging
- `'wandb_project'`: *str*, Name of weights & biases project to log to.
- `'wandb_entity'`: *str*, username or team name for weights and biases.
- `'name'`: *str*, experiment name file-saving, and for weights & biases
- `'notes'`: *str*, experiment notes for weights & biases
- `'sample_epochs'`: *int*, epochs between generating samples.
- `'n_samples'`: *int*, number of samples to generate 
- `'sample_dir'`: *str*, directory to save samples
- `'checkpoint_path'`: *str*, path of saved checkpoints (at best validation likelihood). If unspecified, and using weights and biases, this defaults to `checkpoint.tar` in the WandB run directory. If not using weights and biases, this defaults to `f"./{str(self.config['name']).replace(' ', '_')}_checkpoint.tar"`.
- `'log_interval'`: *int*, number of batches between printing training loss.
- `'verbose'`: *bool*, if True, log the log-jacobian-determinant and reconstruction loss per layer separately to weights and biases.
- `'sample_true_inv'`: *bool*, if True, generate samples from the true inverse of a self-normalizing flow model, in addition to samples from the approximate (fast) inverse.
- `'plot_recon'`: *bool*, if True, plot reconstruction of training images.
- `'log_timing'`: *bool*, if True, compute mean and std. of time per batch and time per sample. Print to screen and save as summary statistic of experiment.

## Running an experiment
### MNIST 

fastflow/fastflow_mnist.py

`set-  -n_blocks=3, block_size=32, image_size=(1, 28, 28)`

      python fastflow_mnist.py
      
### Imagenet 32/64

fastflow/fastflow_imagenet_multi_gpu.py

`set-  resulotion=32/64, -n_blocks=2, block_size=16, image_size=(3, 32, 32)`

      python fastflow_imagenet_multi_gpu.py   
fastflow/fastflow_celeba_multi_gpu.py

`set-  resulotion=32/64/128, -n_blocks=3, block_size=32, image_size=(3, 32, 32)`

      python fastflow_celeba_multi_gpu.py 

# Acknowledgements
The training and testing code in this repo is based on code in Self Normalizing Flows Github(https://github.com/akandykeller/SelfNormalizingFlows). The implementation of FInC Flow Step is based on MaCoW(https://github.com/XuezheMax/macow)

# Citation

Please use the following BibTex Code to cite our paper
```
@conference{visapp23,
author={Aditya Kallappa. and Sandeep Nagar. and Girish Varma.},
title={FInC Flow: Fast and Invertible k × k Convolutions for Normalizing Flows},
booktitle={Proceedings of the 18th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISIGRAPP 2023) - Volume 5: VISAPP},
year={2023},
pages={338-348},
publisher={SciTePress},
organization={INSTICC},
doi={10.5220/0011876600003417},
isbn={978-989-758-634-7},
issn={2184-4321},
}
```


import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"

import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from layers import Dequantization, Normalization
from layers.distributions.uniform import UniformDistribution
from layers.splitprior import SplitPrior
from layers.flowsequential import FlowSequential
from layers.conv1x1 import Conv1x1
from layers.actnorm import ActNorm
from layers.squeeze import Squeeze
from layers.transforms import LogitTransform
from layers.coupling import Coupling
from train.losses import NegativeGaussianLoss
from train.experiment import Experiment
from datasets.cifar10 import load_data

from layers.conv import PaddedConv2d#, Conv1x1
from fastflow import FastFlowUnit
from datetime import datetime

now = datetime.now()
# dd/mm/YY HH/MM/SS
date_time = now.strftime("%d:%m:%Y %H:%M:%S")

lr = 1e-3
optimizer_ = 'Adam'
scheduler_ = 'Exponential_0.99'
multi_gpu = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    multi_gpu = True
def create_model(num_blocks=3, block_size=32, sym_recon_grad=False, 
                 actnorm=False, split_prior=False, recon_loss_weight=1.0):
    current_size = (3, 32, 32)

    alpha = 1e-6
    layers = [
        Dequantization(UniformDistribution(size=current_size)),
        Normalization(translation=0, scale=256),
        Normalization(translation=-alpha, scale=1 / (1 - 2 * alpha)),
        LogitTransform(),
    ]

    for l in range(num_blocks):
        layers.append(Squeeze())
        current_size = (current_size[0]*4, current_size[1]//2, current_size[2]//2)

        for k in range(block_size):
            layers.append(FastFlowUnit(current_size[0], current_size[0], (3, 3)))
            if actnorm:
                layers.append(ActNorm(current_size[0]))
            layers.append(Conv1x1(current_size[0]))
            layers.append(Coupling(current_size))

        if split_prior and l < num_blocks - 1:
            layers.append(SplitPrior(current_size, NegativeGaussianLoss))
            current_size = (current_size[0] // 2, current_size[1], current_size[2])

    return FlowSequential(NegativeGaussianLoss(size=current_size), 
                         *layers)

run_name = f'{optimizer_}_{scheduler_}_{lr}_{date_time}'
def main():
    config = {
        'name': f'3L-32K FastFlow_CIFAR_{run_name}',
        'eval_epochs': 1,
        'sample_epochs': 1,
        'log_interval': 100,
        'lr': lr,
        'num_blocks': 1,
        'block_size': 1,
        'batch_size': 800,
        'modified_grad': False,
        'add_recon_grad': False,
        'sym_recon_grad': False,
        'actnorm': True,
        'split_prior': True,
        'activation': 'None',
        'recon_loss_weight': 0.0,
        'sample_true_inv': False,
        'plot_recon': True,
        'grad_clip_norm': None,
        'dataset': 'CIFAR',
        'run_name': f'{run_name}',
        'wandb_project': 'fast-flow-CIFAR',
        'Optimizer': optimizer_,
        'Scheduler': scheduler_,
        'multi_gpu': multi_gpu
    }

    train_loader, val_loader, test_loader = load_data(data_aug=True, batch_size=config['batch_size'])

    model = create_model(num_blocks=config['num_blocks'],
                         block_size=config['block_size'], 
                         sym_recon_grad=config['sym_recon_grad'],
                         actnorm=config['actnorm'],
                         split_prior=config['split_prior'],
                         recon_loss_weight=config['recon_loss_weight'])#.to('cuda')
    
    if config['multi_gpu']:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, [0, 1], device)
    # print(model)
    model.to(device)
    # print(model)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = ExponentialLR(optimizer, gamma=0.99, last_epoch=-1)

    experiment = Experiment(model, train_loader, val_loader, test_loader,
                            optimizer, scheduler, **config)
    experiment.run()


if __name__ == '__main__':
    main()
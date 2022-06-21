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
from datasets.imagenet import load_data
from fastflow import FastFlowUnit
from datetime import datetime

now = datetime.now()
# dd/mm/YY HH/MM/SS

run_name = now.strftime("%d:%m:%Y %H:%M:%S")
lr = 1e-3
optimizer_ = 'Adam'
scheduler_ = 'Exponential_0.99'
multi_gpu = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model(num_blocks=3, block_size=48, sym_recon_grad=False, 
                 actnorm=False, split_prior=False, recon_loss_weight=1.0):
    current_size = (3, 64, 64)

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
            # layers.append(FastFlowUnit(current_size[0], current_size[0], (3, 3)))
            if actnorm:
                layers.append(ActNorm(current_size[0]))
            layers.append(Conv1x1(current_size[0]))
            layers.append(Coupling(current_size))

        if split_prior and l < num_blocks - 1:
            layers.append(SplitPrior(current_size, NegativeGaussianLoss))
            current_size = (current_size[0] // 2, current_size[1], current_size[2])

    return FlowSequential(NegativeGaussianLoss(size=current_size), 
                         *layers)


def main():
    config = {
        'name': '3L-48K Imagenet64',
        'eval_epochs': 1,
        'sample_epochs': 1,
        'log_interval': 100,
        'lr': lr,
        'num_blocks': 4,
        'block_size': 48,
        'batch_size': 10,
        'modified_grad': False,
        'add_recon_grad': False,
        'sym_recon_grad': False,
        'actnorm': True,
        'split_prior': True,
        'activation': 'None',
        'recon_loss_weight': 0.0,
        'sample_true_inv': False,
        'plot_recon': False,
        'grad_clip_norm': 10_000,
        'warmup_epochs': 0,
        'dataset': 'Imagenet 32',
        'run_name': f'{run_name}',
        'wandb_project': 'fast-flow-Imagenet-Glow',
        'Optimizer': optimizer_,
        'Scheduler': scheduler_,
        'multi_gpu': multi_gpu
    }

    train_loader, val_loader, test_loader = load_data(data_aug=False, resolution=64, 
              data_dir='/scratch/aditya.kallappa/Imagenet', batch_size=config['batch_size'])

    model = create_model(num_blocks=config['num_blocks'],
                         block_size=config['block_size'], 
                         sym_recon_grad=config['sym_recon_grad'],
                         actnorm=config['actnorm'],
                         split_prior=config['split_prior'],
                         recon_loss_weight=config['recon_loss_weight']).to('cuda')

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    # scheduler = StepLR(optimizer, step_size=1, gamma=1.0)
    scheduler = ExponentialLR(optimizer, gamma=0.99, last_epoch=-1)

    experiment = Experiment(model, train_loader, val_loader, test_loader,
                            optimizer, scheduler, **config)

    experiment.run()

if __name__ == '__main__':
    main()
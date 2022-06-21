import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau, ExponentialLR

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
from datasets.mnist import load_data


from layers.conv import PaddedConv2d#, Conv1x1
from cinc_flow import CINCFlowUnit

from datetime import datetime

now = datetime.now()


# dd/mm/YY HH/MM/SS
run_name = now.strftime("%d:%m:%Y %H:%M:%S")
optimizer_ = "Adam"
scheduler_ = "Exponential_0.99"
lr = 1e-3


def create_model(num_blocks=2, block_size=16, sym_recon_grad=False, 
                 actnorm=False, split_prior=False, recon_loss_weight=1.0, image_size=(1, 28, 28)):

    current_size = image_size
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
            layers.append(CINCFlowUnit(current_size[0], current_size[0], (3, 3)))
            if actnorm:
                layers.append(ActNorm(current_size[0]))
            # layers.append(Conv1x1(current_size[0], current_size[0]))
            layers.append(Conv1x1(current_size[0]))
            layers.append(Coupling(current_size))

        if split_prior and l < num_blocks - 1:
            layers.append(SplitPrior(current_size, NegativeGaussianLoss))
            current_size = (current_size[0] // 2, current_size[1], current_size[2])

    return FlowSequential(NegativeGaussianLoss(size=current_size), 
                         *layers)
    # return NegativeGaussianLoss(size=current_size), *layers

def main():
    config = {
        'name': f'2L-16K FastFlow_MNIST_{optimizer_}_{scheduler_}_{lr}_{run_name}',
        'eval_epochs': 1,
        'sample_epochs': 1,
        'log_interval': 100,
        'lr': lr,
        'num_blocks': 2,
        'block_size': 16,
        'batch_size': 250,
        'modified_grad': False,
        'add_recon_grad': False,
        'sym_recon_grad': False,
        'actnorm': True,
        'split_prior': True,
        'activation': 'None',
        'recon_loss_weight': 1.0,
        'sample_true_inv': False,
        'plot_recon': True,
        'dataset': 'MNIST',
        'run_name': f'{run_name}',
        'Optimizer': optimizer_,
        'Scheduler': scheduler_
    }

    train_loader, val_loader, test_loader = load_data(data_aug=False, batch_size=config['batch_size'])

    model = create_model(num_blocks=config['num_blocks'],
                         block_size=config['block_size'], 
                         sym_recon_grad=config['sym_recon_grad'],
                         actnorm=config['actnorm'],
                         split_prior=config['split_prior'],
                         recon_loss_weight=config['recon_loss_weight']).to('cuda')
    print(model.parameters())
    # optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    # optimizer = optim.Adamax(model.parameters(), lr=config['lr'])
    # optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=0.01)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = None
    # scheduler = MultiStepLR(optimizer, milestones=[50, 100, 200, 500], gamma=0.5)   
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, threshold=1.0) 
    scheduler = ExponentialLR(optimizer, gamma=0.99, last_epoch=-1)
    experiment = Experiment(model, train_loader, val_loader, test_loader,
                            optimizer, scheduler, **config)

    experiment.run()

if __name__ == '__main__':
    main()
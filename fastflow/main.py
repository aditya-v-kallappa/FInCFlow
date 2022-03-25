
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from torch import optim
from torch.optim.lr_scheduler import StepLR

from layers import Dequantization, Normalization
from layers.distributions.uniform import UniformDistribution
from layers.splitprior import SplitPrior
from layers.flowsequential import FlowSequential
# from layers.conv1x1 import Conv1x1
from layers.actnorm import ActNorm
from layers.squeeze import Squeeze
from layers.transforms import LogitTransform
from layers.coupling import Coupling
from train.losses import NegativeGaussianLoss
from train.experiment import Experiment
from datasets.mnist import load_data


from conv import PaddedConv2d, Conv1x1
from fastflow import FastFlowUnit

from model_mnist import create_model

config = {
        'name': '2L-16K Glow Exact MNIST',
        'eval_epochs': 1,
        'sample_epochs': 1,
        'log_interval': 100,
        'lr': 1e-3,
        'num_blocks': 2,
        'block_size': 16,
        'batch_size': 100,
        'modified_grad': False,
        'add_recon_grad': False,
        'sym_recon_grad': False,
        'actnorm': True,
        'split_prior': True,
        'activation': 'None',
        'recon_loss_weight': 1.0,
        'sample_true_inv': False,
        'plot_recon': False
    }



class LitAutoEncoder(pl.LightningModule):
	def __init__(self):
		super().__init__()
        self.distribution, modules = model = create_model(  num_blocks=config['num_blocks'],
                                                            block_size=config['block_size'], 
                                                            sym_recon_grad=config['sym_recon_grad'],
                                                            actnorm=config['actnorm'],
                                                            split_prior=config['split_prior'],
                                                            recon_loss_weight=config['recon_loss_weight']).to('cuda')
        for i, module in enumerate(modules):
            self.add_module(str(i), module)
        self.sequence_modules = modules
        
	def forward(self, x):
		embedding = self.encoder(x)
		return embedding

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		x, y = train_batch
		x = x.view(x.size(0), -1)
		z = self.encoder(x)    
		x_hat = self.decoder(z)
		loss = F.mse_loss(x_hat, x)
		self.log('train_loss', loss)
		return loss

	def validation_step(self, val_batch, batch_idx):
		x, y = val_batch
		x = x.view(x.size(0), -1)
		z = self.encoder(x)
		x_hat = self.decoder(z)
		loss = F.mse_loss(x_hat, x)
		self.log('val_loss', loss)

# # data
# dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
# mnist_train, mnist_val = random_split(dataset, [55000, 5000])

# train_loader = DataLoader(mnist_train, batch_size=32)
# val_loader = DataLoader(mnist_val, batch_size=32)

# # model
# model = LitAutoEncoder()

# # training
# trainer = pl.Trainer(gpus=4, num_nodes=8, precision=16, limit_train_batches=0.5)
# trainer.fit(model, train_loader, val_loader)
    

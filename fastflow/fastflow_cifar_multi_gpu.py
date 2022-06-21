import math
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau, ExponentialLR

from layers import Dequantization, Normalization
from layers.distributions.uniform import UniformDistribution
# from layers.splitprior import SplitPrior
from layers.flowsequential import FlowSequential
from layers.conv1x1 import Conv1x1
from layers.actnorm import ActNorm
from layers.squeeze import Squeeze
from layers.transforms import LogitTransform
from layers.coupling import Coupling
from train.losses import NegativeGaussianLoss
from train.experiment import Experiment
from datasets.mnist import load_data as load_data_mnist
from datasets.cifar10 import load_data as load_data_cifar
from layers.flowlayer import FlowLayer
from train.experiment import Experiment
from layers.conv import PaddedConv2d#, Conv1x1
from fastflow import FastFlowUnit

from collections import OrderedDict
from datetime import datetime


def clear_grad(module):
    if isinstance(module, PaddedConv2d):
        # print("_____Before______________")
        # print(module.order)
        # print(module.conv.weight.data)
        # print(module.conv.weight.grad)
        module.reset_gradients() 
        # print("______After_____________")
        # print(module.order)
        # print(module.conv.weight.data)
        # print(module.conv.weight.grad)

class SplitPrior(FlowLayer):
    def __init__(self, size, distribution, width=512):
        super().__init__()
        assert len(size) == 3
        self.n_channels = size[0]

        self.transform = Coupling(size, width=width)

        self.base = distribution(
            (self.n_channels // 2, size[1], size[2]))

    def forward(self, input, context=None):
        x, ldj = self.transform(input, context)

        x1 = x[:, :self.n_channels // 2, :, :]
        x2 = x[:, self.n_channels // 2:, :, :]

        log_pz2 = self.base.log_prob(x2)
        log_px2 = log_pz2 + ldj

        return x1, x2, log_px2

    def reverse(self, input, x2=None, context=None):
        x1 = input
        if x2 is None:
            x2, log_px2 = self.base.sample(x1.shape[0], context)

        x = torch.cat([x1, x2], dim=1)
        x = self.transform.reverse(x, context)

        return x

    def logdet(self, input, context=None):
        x1, ldj = self.forward(input, context)
        return ldj

class Split(FlowLayer):
    """ Split layer; cf Glow figure 2 / RealNVP figure 4b
    Based on RealNVP multi-scale architecture: splits an input in half along the channel dim; half the vars are
    directly modeled as Gaussians while the other half undergo further transformations (cf RealNVP figure 4b).
    """
    def __init__(self, size, distribution):
        super().__init__()
        self.n_channels = size[0]
        self.gaussianize = Gaussianize(self.n_channels//2)
        self.base = distribution(size=(self.n_channels // 2, size[1], size[2]))

    def forward(self, x, context=None):
        x1, x2 = x.chunk(2, dim=1)  # split input along channel dim
        z2, logdet = self.gaussianize(x1, x2)
        log_pz2 = self.base.log_prob(z2)
        logdet += log_pz2
        return x1, z2, logdet
        # return x1, logdet

    def reverse(self, x1, z2=None, context=None):#, z2):
        if z2 is None:
            z2, log_pz2 = self.base.sample(x1.shape[0], context)
        x2 = self.gaussianize.reverse(x1, z2)
        x = torch.cat([x1, x2], dim=1)  # cat along channel dim
        return x

    def logdet(self, x, context=None):
        x, logdet = self.forward(x, context)
        return logdet
    
    def reconstruct_forward(self, x):
        x1, x2 = x.chunk(2, dim=1)  # split input along channel dim
        z2, logdet = self.gaussianize(x1, x2)
        log_pz2 = self.base.log_prob(z2)
        logdet += log_pz2
        self.z2 = z2.detach()
        # return x1, z2, logdet
        return x1, logdet
    
    def reconstruct_reverse(self, x1, z2=None):
        # assert z2, "Provide the z values or call the reverse function"
        if z2 is None:
            z2 = self.base.sample(x1.shape[0])
        else:
            z2 = self.z2
        x2, logdet = self.gaussianize.reverse(x1, z2)
        x = torch.cat([x1, x2], dim=1)  # cat along channel dim
        return x

class Gaussianize(nn.Module):
    """ Gaussianization per ReanNVP sec 3.6 / fig 4b -- at each step half the variables are directly modeled as Gaussians.
    Model as Gaussians:
        x2 = z2 * exp(logs) + mu, so x2 ~ N(mu, exp(logs)^2) where mu, logs = f(x1)
    then to recover the random numbers z driving the model:
        z2 = (x2 - mu) * exp(-logs)
    Here f(x1) is a conv layer initialized to identity.
    """
    def __init__(self, n_channels):
        super().__init__()
        self.net = nn.Conv2d(n_channels, 2*n_channels, kernel_size=3, padding=1)  # computes the parameters of Gaussian
        self.log_scale_factor = nn.Parameter(torch.zeros(2*n_channels,1,1))       # learned scale (cf RealNVP sec 4.1 / Glow official code
        # initialize to identity
        self.net.weight.data.zero_()
        self.net.bias.data.zero_()

    def forward(self, x1, x2=None):
        if x2 is None:
            x1, x2 = x1.chunk(2, dim=1)
        h = self.net(x1) * self.log_scale_factor.exp()  # use x1 to model x2 as Gaussians; learnable scale
        m, logs = h[:,0::2,:,:], h[:,1::2,:,:]          # split along channel dims
        z2 = (x2 - m) * torch.exp(-logs)                # center and scale; log prob is computed at the model forward
        logdet = - logs.sum([1,2,3])
        return z2, logdet

    def reverse(self, x1, z2):
        # if z2 is None:
        #     z2, log_pz2 = self.base.sample(x1.shape[0], context)
        # else:
        #     log_pz2 = 0.0
        h = self.net(x1) * self.log_scale_factor.exp()
        m, logs = h[:,0::2,:,:], h[:,1::2,:,:]
        x2 = m + z2 * torch.exp(logs)
        # logdet = logs.sum([1,2,3])
        return x2

class Preprocess(nn.Module):
    def __init__(self, size):
        super().__init__()
        alpha = 1e-6
        layers = [
            Dequantization(UniformDistribution(size=size)),
            Normalization(translation=0, scale=256),
            Normalization(translation=-alpha, scale=1 / (1 - 2 * alpha)),
            LogitTransform()
        ]
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        logdet = 0
        for layer in self.layers:
            x, layer_logdet = layer(x)#, context=None)
            logdet += layer_logdet

        return x, logdet
    
    def reverse(self, x):
        for layer in reversed(self.layers):
            x = layer.reverse(x)#, context=None)
        
        return x


class GlowStep(nn.Module):
    def __init__(self, size, actnorm=False):
        super().__init__()
        dict = OrderedDict()
        if actnorm:
            # layers.append(ActNorm(size[0]))
            actnorm = ActNorm(size[0])
            # self.glow_step.append(actnorm)
            dict['actnorm'] = actnorm
        # layers.append(Conv1x1(size[0]))
        # layers.append(Coupling(size))
        conv1x1 = Conv1x1(size[0])
        coupling = Coupling(size)
        dict['conv1x1'] = conv1x1
        dict['coupling'] = coupling
        # self.glow_step.append(conv)
        # self.glow_step.append(coupling)
        self.glow_step = nn.Sequential(dict)
        
    
    def forward(self, x):
        logdet = 0
        for layer in self.glow_step:
            x, layer_logdet = layer(x)#, context=None)
            logdet += layer_logdet

        return x, logdet
    
    def reverse(self, x):
        for layer in reversed(self.glow_step):
            # print("Inside GLowStep", layer)
            x = layer.reverse(x)#, context=None)
        
        return x

class FastFlowStep(nn.Module):
    def __init__(self, size, actnorm=False):
        super().__init__()
        fastflow_unit = FastFlowUnit(size[0], size[0], (3, 3))
        glow_unit = GlowStep(size, actnorm)
        dict = OrderedDict([
          ('fastflow_unit', fastflow_unit),
          ('glow_unit', glow_unit)
        ])
        # self.fastflow_step = nn.ModuleList({
        #     'fastflow_unit': fastflow_unit,
        #     'glow_unit': glow_unit
        # })
        self.fastflow_step = nn.Sequential(dict)
    def forward(self, x):
        logdet = 0
        # x, layer_logdet = self.fastflow_unit(x, context=None)
        # logdet += layer_logdet
        # x, layer_logdet = self.glow_unit(x, context=None)
        # logdet += layer_logdet
        for layer in self.fastflow_step:
            x, layer_logdet = layer(x)#, context=None)
            logdet += layer_logdet

        return x, logdet    
    
    def reverse(self, x):
        # print("In FastFlow Step Inverse")
        for layer in reversed(self.fastflow_step):
            # print(layer)
            x = layer.reverse(x)#, context=None)

        return x


class FastFlowLevel(nn.Module):
    def __init__(self, size, block_size=16, actnorm=False):
        super().__init__()
        squeeze = Squeeze()
        size = (size[0] * 4, size[1] // 2, size[2] // 2)
        split = SplitPrior(size, NegativeGaussianLoss)
        # split = Split(size, NegativeGaussianLoss)
        self.fastflow_level = nn.ModuleList([
            squeeze,
            *[FastFlowStep(size, actnorm) for _ in range(block_size)],
            split
        ])
    def forward(self, x):
        logdet = 0
        # x, layer_logdet = self.fastflow_unit(x, context=None)
        # logdet += layer_logdet
        # x, layer_logdet = self.glow_unit(x, context=None)
        # logdet += layer_logdet
        for layer in self.fastflow_level:
            if not isinstance(layer, SplitPrior):
                x, layer_logdet = layer(x)
            else:
                x, z, layer_logdet = layer(x)
            logdet += layer_logdet
        return x, z, logdet    
    
    def reverse(self, x, z=None):
        for layer in reversed(self.fastflow_level):
            # print("In FastFlowLevel Reverse")
            # print(layer)
            if not isinstance(layer, SplitPrior):
                x = layer.reverse(x)#, context=None)
            else:
                x = layer.reverse(x, z)
        return x

class FastFlow(nn.Module):
    def __init__(self, n_blocks=2, block_size=16, image_size=(1, 28, 28), actnorm=False):
        super(FastFlow, self).__init__()
        size = image_size
        C_in, H, W = size
        C_out = C_in * (2 ** (n_blocks + 1))
        H_out = H // (2 ** n_blocks)
        W_out = W // (2 ** n_blocks)
        
        n_levels = n_blocks - 1

        self.output_size = (C_out, H_out, W_out)

        self.preprocess = Preprocess(size=size)
        self.fastflow_levels = nn.ModuleList([FastFlowLevel((C_in * (2**i), H//(2**i), W//(2**i)), block_size, actnorm) for i in range(n_levels)])
        self.squeeze = Squeeze()
        self.fastflow_step = nn.Sequential(*[FastFlowStep(self.output_size, actnorm) for _ in range(block_size)])
        # self.gaussianize = Gaussianize(C_out)
        self.base_distribution = NegativeGaussianLoss(size=self.output_size)

    def forward(self, x, context=None):
        logdet = 0
        zs = []
        x, layer_logdet = self.preprocess(x)
        logdet += layer_logdet
        # print("After preprocess", x.shape, torch.cuda.current_device())
        for module in self.fastflow_levels:
            x, z, layer_logdet = module(x)
            logdet += layer_logdet
            zs.append(z)
        # print("After fastflow_levels", x.shape, torch.cuda.current_device())
        x, layer_logdet = self.squeeze(x)
        logdet += layer_logdet
        # print("Outside Squeeze:", x.shape, torch.cuda.current_device())
        for module in self.fastflow_step:
            x, layer_logdet = module(x)
            logdet += layer_logdet
        
        # print("After Outside step", x.shape, torch.cuda.current_device())
        # x, layer_logdet = self.gaussianize(torch.zeros_like(x), x)
        # logdet += layer_logdet     
        logdet += self.base_distribution.log_prob(x)     
        zs.append(x)
        # print("Output:", x.shape, torch.cuda.current_device())
        return zs, logdet#x, logdet
    
    def reverse(self, n_samples=1, zs=None, z_std=1.0):
        if zs is None:  # if no random numbers are passed, generate new from the base distribution
            assert n_samples is not None, 'Must either specify n_samples or pass a batch of z random numbers.'
            # zs = [z_std * self.base_distribution.sample(n_samples).squeeze()]
            zs = self.base_distribution.sample(n_samples)[0]
            zs = [zs]
        
        # z = self.gaussianize.reverse(torch.zeros_like(zs[-1]),zs[-1])
        # print("After inverse gaussianize", z.shape)
        z = zs[-1]
        for module in reversed(self.fastflow_step):
            z = module.reverse(z)
        x = self.squeeze.reverse(z)

        for i, m in enumerate(reversed(self.fastflow_levels)):
            
            # z = z_std * (self.base_dist.sample(x.shape).squeeze() if len(zs)==1 else zs[-i-2])  # if no z's are passed, generate new random numbers from the base dist
            if len(zs) == 1:
                x = m.reverse(x)#, z)
            else:
                x = m.reverse(x, zs[-i-2])

            
        # postprocess
        x = self.preprocess.reverse(x)
        
        return x

    def log_prob(self, x, bits_per_pixel=False):
        x, log_prob = self.forward(x)
        if bits_per_pixel:
            log_prob /= (math.log(2) * x[0].numel()) 
        return log_prob

    def sample(self, n_samples, context=None, compute_expensive=False, 
                also_true_inverse=False):
        z, logprob = self.base_distribution.sample(n_samples, context)
        z = [z]
        # Regular sample
        input = z
        x = self.reverse(n_samples=n_samples, zs=z)
        return x, x #true_x

    def reconstruct(self, x, context=None, compute_expensive=False):
        zs, _ = self.forward(x)
        x = self.reverse(n_samples=x.shape[0], zs=[zs[-1]])   
        return x


# 
        
if __name__ == '__main__':

    now = datetime.now()
    # dd/mm/YY HH/MM/SS
    date_time = now.strftime("%d:%m:%Y %H:%M:%S")
    lr = 1e-3
    optimizer_ = 'Rprop'
    scheduler_ = 'Exponential_0.99997'
    multi_gpu = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        multi_gpu = True
    run_name = f'{optimizer_}_{scheduler_}_{lr}_{date_time}'
    config = {
        'name': f'3L-32K FastFlow_CIFAR_{run_name}',
        'eval_epochs': 1,
        'sample_epochs': 1,
        'log_interval': 100,
        'lr': lr,
        'num_blocks': 3,
        'block_size': 32,
        'batch_size': 400,
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
        'wandb_project': 'fast-flow-CIFAR-Matched',
        'Optimizer': optimizer_,
        'Scheduler': scheduler_,
        'multi_gpu': multi_gpu,
        'loss_bpd': False
    }

    train_loader, val_loader, test_loader = load_data_cifar(data_aug=True, batch_size=config['batch_size'])

    model = FastFlow(n_blocks=config['num_blocks'],
                     block_size=config['block_size'],
                     actnorm=config['actnorm'],
                     image_size=(3, 32, 32))#.to("cuda")
    if config['multi_gpu']:
        model = nn.DataParallel(model)
    
    model = model.to('cuda')
    
    # optimizer = optim.Adam(model.parameters(), lr=config['lr'])#, weight_decay=1e-5)
    optimizer = optim.Rprop(model.parameters(), lr=config['lr'])#, weight_decay=1e-5)
    # scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
    scheduler = ExponentialLR(optimizer, gamma=0.99997, last_epoch=-1)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, threshold=10) 
    experiment = Experiment(model, train_loader, val_loader, test_loader,
                            optimizer, scheduler, **config)
    experiment.run()




# def main():
#     model = FastFlow(n_blocks=2, block_size=1, actnorm=True)#.to("cuda")
#     model = nn.DataParallel(model)
#     model = model.to('cuda')
#     # print(model)
#     fastflow_params = []
#     glow_params = []
#     # for name, param in model.named_parameters():
#     #     if 'fastflow_unit' in name:
#     #         fastflow_params.append(param)
#     #     else:
#     #         glow_params.append(param)
#     # print(fastflow_params)
#     # print(glow_params)

#     # for m in model.parameters():
#     #     print(m)

#     # optimizer = optim.Adam([
#     #     {'params': fastflow_params, 'lr': 1e-6},
#     #     {'params': glow_params}
#     # ], lr=1e-6)

#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
#     # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
#     scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, last_epoch=-1)

#     train_loader, val_loader, test_loader = load_data(data_aug=False, batch_size=1250)
#     for e in range(50):
#         avg_loss = train(model, train_loader, optimizer, scheduler)
#         print(e, avg_loss)
#         scheduler.step()

# def train(model, train_loader, optimizer, scheduler):
#     batches = 0
#     total_loss = 0
#     for input, label in train_loader:
#         optimizer.zero_grad()
#         input = input.to("cuda")
#         # out, _ = model(input)
#         # loss = -model.log_prob(input, bits_per_pixel=True).mean(0)
#         _, loss = model.forward(input) 
#         loss = -loss.mean(0)
#         # print(loss)
#         loss.backward()
#         # print(loss)
#         model.apply(clear_grad)
#         # print("+=========================================================================")
#         optimizer.step()
#         total_loss += loss
#         batches += 1
    
#     return total_loss/batches
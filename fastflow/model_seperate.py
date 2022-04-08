import math
import torch
import torch.nn as nn
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
from layers.flowlayer import FlowLayer

from layers.conv import PaddedConv2d#, Conv1x1
from fastflow import FastFlowUnit

from collections import OrderedDict


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
        # return x1, z2, logdet
        return x1, logdet

    def reverse(self, x1, context=None):#, z2):
        z2, log_pz2 = self.base.sample(x1.shape[0], context)
        x2, logdet = self.gaussianize.reverse(x1, z2)
        x = torch.cat([x1, x2], dim=1)  # cat along channel dim
        return x, logdet
    
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
        return x, logdet

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

    def reverse(self, x1, z2=None):
        if z2 is not None:
            z2, log_pz2 = self.base.sample(x1.shape[0], context)
        else:
            log_pz2 = 0.0
        h = self.net(x1) * self.log_scale_factor.exp()
        m, logs = h[:,0::2,:,:], h[:,1::2,:,:]
        x2 = m + z2 * torch.exp(logs)
        logdet = logs.sum([1,2,3])
        return x2, logdet + log_pz2

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
        for layer in reversed(self.fastflow_step):
                x = layer.reverse(x)#, context=None)

        return x


class FastFlowLevel(nn.Module):
    def __init__(self, size, block_size=16, actnorm=False):
        super().__init__()
        squeeze = Squeeze()
        size = (size[0] * 4, size[1] // 2, size[2] // 2)
        # split = SplitPrior(size, NegativeGaussianLoss)
        split = Split(size, NegativeGaussianLoss)
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
            x, layer_logdet = layer(x)
            logdet += layer_logdet

        return x, logdet    
    
    def reverse(self, x):
        for layer in reversed(self.fastflow_level):
                x = layer.reverse(x)#, context=None)

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
        self.fastflow_step = FastFlowStep(self.output_size, actnorm)
        self.gaussianize = Gaussianize(C_out)
        self.base_distribution = NegativeGaussianLoss(size=self.output_size)

    def forward(self, x, context=None):
        logdet = 0

        x, layer_logdet = self.preprocess(x)
        logdet += layer_logdet
        for module in self.fastflow_levels:
            x, layer_logdet = module(x)
            logdet += layer_logdet

        x, layer_logdet = self.squeeze(x)
        logdet += layer_logdet

        x, layer_logdet = self.fastflow_step(x)
        logdet += layer_logdet
        
        x, layer_logdet = self.gaussianize(torch.zeros_like(x), x)
        logdet += layer_logdet          
        return x, logdet
    
    def reverse(self, n_samples=1, zs=None, z_std=1.0):
        if zs is None:  # if no random numbers are passed, generate new from the base distribution
            assert n_samples is not None, 'Must either specify n_samples or pass a batch of z random numbers.'
            zs = [z_std * self.base_distribution.sample(n_samples).squeeze()]
        logdet = 0
        z, layer_logdet = self.gaussianize.reverse(torch.zeros_like(zs[-1]), zs[-1])
        logdet += layer_logdet

        x, layer_logdet = self.fastflow_step.reverse(z)
        logdet += layer_logdet

        x, layer_logdet = self.squeeze.reverse(x)
        logdet += layer_logdet

        for i, m in enumerate(reversed(self.fastflow_levels)):
            # z = z_std * (self.base_dist.sample(x.shape).squeeze() if len(zs)==1 else zs[-i-2])  # if no z's are passed, generate new random numbers from the base dist
            x, layer_logdet = m.reverse(x)#, z)

            logdet += layer_logdet
        # postprocess
        x, layer_logdet = self.preprocess.reverse(x)
        logdet += layer_logdet
        return x, logdet

    def log_prob(self, x, bits_per_pixel=False):
        x, log_prob = self.forward(x)
        if bits_per_pixel:
            log_prob /= (math.log(2) * x[0].numel()) 
        return log_prob

def main():
    model = FastFlow(n_blocks=2, block_size=16, actnorm=True).to("cuda")
    # print(model)
    fastflow_params = []
    glow_params = []
    for name, param in model.named_parameters():
        if 'fastflow_unit' in name:
            fastflow_params.append(param)
        else:
            glow_params.append(param)
    # print(fastflow_params)
    # print(glow_params)

    # for m in model.parameters():
    #     print(m)

    optimizer = optim.Adam([
        {'params': fastflow_params, 'lr': 1e-6},
        {'params': glow_params}
    ], lr=1e-6)

    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_loader, val_loader, test_loader = load_data(data_aug=False, batch_size=250)
    for e in range(10):
        avg_loss = train(model, train_loader, optimizer, scheduler)
        print(e, avg_loss)

def train(model, train_loader, optimizer, scheduler):
    batches = 0
    total_loss = 0
    for input, label in train_loader:
        optimizer.zero_grad()
        input = input.to("cuda")
        # out, _ = model(input)
        loss = -model.log_prob(input, bits_per_pixel=True).mean(0)
        loss.backward()
        # print(loss)
        model.apply(clear_grad)
        # print("+=========================================================================")
        optimizer.step()
        total_loss += loss
        batches += 1
    
    return total_loss/batches
        
if __name__ == '__main__':
    main()
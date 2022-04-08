import torch
from layers.conv import PaddedConv2d
from utils.solve_mc import solve
import time
from fastflow import FastFlowUnit
import matplotlib.pyplot as plt
import numpy as np
from layers.split import Split

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
from layers.flowlayer import ModifiedGradFlowLayer
from fastflow_mnist import create_model as create_model_fastflow
from glow_mnist import create_model as create_model_glow

def test_PaddedConv2d(x=None, w=None, kernel_size=(3, 3), bias=False, order='TL', is_input=True, print_answer=False):
    if x is None:
        x = torch.randn((1, 2, 3, 3))
    B, C, H, W = x.shape
    if w is None:
        K_H, K_W = kernel_size[0], kernel_size[1]
    else:
        C_in, C_out, K_H, K_W = w.shape
        assert C_in == C_out and C_in == C
    layer = PaddedConv2d(C, C, (K_H, K_W), bias=bias, order=order)
    layer.conv.weight.data = w if w is not None else layer.conv.weight.data
    layer.eval()
    with torch.no_grad():
        if is_input:
            forward_output, _ = layer.forward(x)
            start_time = time.time()
            reverse_input, _ = layer.reverse(forward_output)
            end_time = time.time()
            print("Error:", torch.mean((x - reverse_input) ** 2).item())
            print("Reverse Time:", end_time - start_time)
            answer = forward_output

        else:
            start_time = time.time()
            reverse_input, _ = layer.reverse(x)
            end_time = time.time()
            forward_output, _ = layer.forward(reverse_input)
            print("Error:", torch.mean((x - forward_output) ** 2).item())
            print("Reverse Time:", end_time - start_time)
            answer = reverse_input
        if print_answer:
            print(answer)

def test_clear_grad(module=None, n_channels=1, order='TL', kernel_size=(3, 3)):
    # K_H, K_W = kernel_size[0], kernel_size[1]
    C = n_channels
    if module is None:
        module = PaddedConv2d(C, C, kernel_size, bias=False, order=order)
    if isinstance(module, PaddedConv2d):
        if module.conv.weight.grad is None:
            module.conv.weight.grad = torch.rand(size=module.conv.weight.data.shape)
        print("Gradients before resetting:\n", module.conv.weight.grad)
        module.reset_gradients() 
        print("Gradients after resetting:\n", module.conv.weight.grad)

def test_FastFlowUnit(x=None, block_size=1, kernel_size=(3, 3), is_input=True, print_answer=False):
    if x is None:
        x = torch.randn((1, 2, 3, 3))
    B, C, H, W = x.shape
    input = x
    k_H, k_W = kernel_size[0], kernel_size[1]
    layers = []
    for _ in range(block_size):
        layers.append(FastFlowUnit(C, C, kernel_size))
    layers = torch.nn.Sequential(*layers)
    # print(layers)
    layers.eval()
    with torch.no_grad():
        if is_input:
            for layer in layers:
                forward_output, _ = layer.forward(x)
                x = forward_output
            total_time = 0
            for layer in reversed(layers):
                start_time = time.time()
                reverse_input = layer.reverse(x)
                x = reverse_input
                end_time = time.time()
                total_time += end_time - start_time
            print("Error:", torch.mean((input - reverse_input) ** 2).item())
            print("Reverse Time:", total_time)
            answer = reverse_input
        else:
            start_time = time.time()
            reverse_input = layers.reverse(x)
            end_time = time.time()
            forward_output, _ = layers.forward(reverse_input)
            print("Error:", torch.mean((x - forward_output) ** 2).item())
            print("Reverse Time:", end_time - start_time)
            answer = forward_output
        if print_answer:
            print(answer)  


def test_Dequantization(x, size, is_input=True):
    layer = Dequantization(UniformDistribution(size=size))
    if is_input:
        forward_output, _ = layer.forward(x)
        reverse_input = layer.reverse(forward_output)
        print("Error:", torch.mean((x - reverse_input) ** 2).item())
    else:
        reverse_input = layer.reverse(x)
        forward_output, _ = layer.forward(reverse_input)
        print("Error:", torch.mean((x - forward_output) ** 2).item())
    

def test_Normalization(x, translation=0, scale=256, is_input=True):
    layer = Normalization(translation, scale)
    if is_input:
        forward_output, _ = layer.forward(x)
        reverse_input = layer.reverse(forward_output)
        print("Error:", torch.mean((x - reverse_input) ** 2).item())
    else:
        reverse_input = layer.reverse(x)
        forward_output, _ = layer.forward(reverse_input)
        print("Error:", torch.mean((x - forward_output) ** 2).item())

def test_LogitTransform(x, is_input=False):
    layer = LogitTransform()
    if is_input:
        forward_output, _ = layer.forward(x)
        reverse_input = layer.reverse(forward_output)
        print("Error:", torch.mean((x - reverse_input) ** 2).item())
    else:
        reverse_input = layer.reverse(x)
        forward_output, _ = layer.forward(reverse_input)
        print("Error:", torch.mean((x - forward_output) ** 2).item())
    
def test_Squeeze(x, is_input=True):
    layer = Squeeze()
    if is_input:
        forward_output, _ = layer.forward(x)
        reverse_input = layer.reverse(forward_output)
        print("Error:", torch.mean((x - reverse_input) ** 2).item())
    else:
        reverse_input = layer.reverse(x)
        forward_output, _ = layer.forward(reverse_input)
        print("Error:", torch.mean((x - forward_output) ** 2).item())
    

def test_ActNorm(x, is_input=True):
    C = x.shape[1]
    layer = ActNorm(C)
    if is_input:
        forward_output, _ = layer.forward(x)
        reverse_input = layer.reverse(forward_output)
        print("Error:", torch.mean((x - reverse_input) ** 2).item())
    else:
        reverse_input = layer.reverse(x)
        forward_output, _ = layer.forward(reverse_input)
        print("Error:", torch.mean((x - forward_output) ** 2).item())


def test_Conv1x1(x, is_input=True):
    C = x.shape[1]
    layer = Conv1x1(C)
    if is_input:
        forward_output, _ = layer.forward(x)
        reverse_input = layer.reverse(forward_output)
        print("Error:", torch.mean((x - reverse_input) ** 2).item())
    else:
        reverse_input = layer.reverse(x)
        forward_output, _ = layer.forward(reverse_input)
        print("Error:", torch.mean((x - forward_output) ** 2).item())

def test_Coupling(x, is_input=True):
    B, C, H, W = x.shape
    layer = Coupling(input_size=(C, H, W))
    if is_input:
        forward_output, _ = layer.forward(x)
        reverse_input = layer.reverse(forward_output)
        print("Error:", torch.mean((x - reverse_input) ** 2).item())
    else:
        reverse_input = layer.reverse(x)
        forward_output, _ = layer.forward(reverse_input)
        print("Error:", torch.mean((x - forward_output) ** 2).item())

def test_SplitPrior(x, is_input=True, print_answer=False):
    x = x.cuda()
    B, C, H, W = x.shape
    layer = SplitPrior(input_size=(C, H, W), distribution=NegativeGaussianLoss).to("cuda")
    if is_input:
        forward_output, _ = layer.forward(x)
        reverse_input = layer.reverse(forward_output)
        print("Error:", torch.mean((x - reverse_input) ** 2).item())
        
        answer = (forward_output, reverse_input)
    else:
        reverse_input = layer.reverse(x)
        forward_output, _ = layer.forward(reverse_input)
        print("Error:", torch.mean((x - forward_output) ** 2).item())
        answer = (reverse_input, forward_output)
    if print_answer:
        print("Input:\n", x)
        print("Output:\n", answer[0], "\n", answer[1])


def test_Split(x, is_input=True, print_answer=False):
    x = x.to("cuda")
    B, C, H, W = x.shape
    layer = Split(C).to("cuda")
    if is_input:
        x1, z2, _ = layer.forward(x)
        reverse_input, _ = layer.inverse(x1, z2)
        print("Error:", torch.mean((x - reverse_input) ** 2).item())
        
        # answer = (forward_output, reverse_input)
    else:
        x1, _ = x[:, :C//2], x[:, C//2:]
        # dist = NegativeGaussianLoss(size=(C, H, W))
        # z2 = dist.sample(n_samples=B)
        z2 = torch.randn(size=(x1.shape)).to(x1.device)
        reverse_input, _ = layer.inverse(x1, z2)
        _x1, _z2, _ = layer.forward(reverse_input)
        forward_output = torch.cat([_x1, _z2], dim=1)
        print("Error:", torch.mean((x - forward_output) ** 2).item())
        answer = (reverse_input, forward_output)
    if print_answer:
        print("Input:\n", x)
        print("Output:\n", answer[0], "\n", answer[1])


def test_FastFlowMNIST(model=None, checkpoint_path=None, x=None, batch_size=1, plot=True):
    if model is None:
        model = create_model_fastflow(num_blocks=2,
                            block_size=16, 
                            sym_recon_grad=False,
                            actnorm=True,
                            split_prior=True,
                            recon_loss_weight=1.0).to('cuda')
    if checkpoint_path:
        print("Loading from ", checkpoint_path)

        checkpoint = torch.load(checkpoint_path)
        summary = checkpoint['summary']
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


    if not x:
        train_loader, val_loader, test_loader = load_data(data_aug=False, batch_size=batch_size)
        for x, label in test_loader:
            x = x.cuda()
            break
    else:
        x = x.cuda()
    
    model.eval()

    reconstructed_image = model.reconstruct(x)
    sample1, _ = model.sample(n_samples=1, also_true_inverse=False)
    if plot:
        fig, ax = plt.subplots(4, 4)
        # print(x)
        # print(reconstructed_image)
        # print(sample1)
        # print(sample2)

        # ax[0, 0].imshow(np.asarray(x.squeeze().detach().cpu().numpy(), dtype=np.int8), cmap='gray')
        # ax[0, 1].imshow(np.asarray(reconstructed_image.squeeze().detach().cpu().numpy(), dtype=np.int8), cmap='gray')
        # ax[1, 0].imshow(np.asarray(sample1[0].squeeze().detach().cpu().numpy(), dtype=np.int8), cmap='gray')
        # ax[1, 1].imshow(np.asarray(sample1[0].squeeze().detach().cpu().numpy(), dtype=np.int8), cmap='gray')
        k = 0
        m = 2
        for i in range(4):
            for j in range(4):
                if i == 0 and j == 0:
                    ax[0, j].imshow(np.asarray(x[m].squeeze().detach().cpu().numpy(), dtype=np.int8), cmap='gray')
                elif i == 0 and j == 1:
                    ax[0, j].imshow(np.asarray(reconstructed_image[m].squeeze().detach().cpu().numpy(), dtype=np.int8), cmap='gray')

                else:
                    ax[i, j].imshow(np.asarray(sample1[k].squeeze().detach().cpu().numpy(), dtype=np.int8), cmap='gray')
                    k += 1
        plt.savefig("image_fastflow.png")
    
    else:
        print(sample1)
        forward_sample = model.forward(sample1)
        
        print("Error:", torch.mean((sample1 - forward_sample) ** 2).item())


def test_GlowMNIST(model=None, checkpoint_path=None, x=None, batch_size=1):
    if model is None:
        model = create_model_glow(num_blocks=2,
                            block_size=16, 
                            sym_recon_grad=False,
                            actnorm=True,
                            split_prior=True,
                            recon_loss_weight=1.0).to('cuda')
    # print(model)
    if checkpoint_path:
        print("Loading from ", checkpoint_path)

        checkpoint = torch.load(checkpoint_path)
        summary = checkpoint['summary']
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


    if not x:
        train_loader, val_loader, test_loader = load_data(data_aug=False, batch_size=batch_size)
        for x, label in test_loader:
            x = x.cuda()
            break
    else:
        x = x.cuda()
    
    model.eval()

    reconstructed_image = model.reconstruct(x)
    sample1, _ = model.sample(n_samples=14, also_true_inverse=False)
    
    fig, ax = plt.subplots(4, 4)
    # print(x)
    # print(reconstructed_image)
    # print(sample1)
    # print(sample2)

    # ax[0, 0].imshow(np.asarray(x.squeeze().detach().cpu().numpy(), dtype=np.int8), cmap='gray')
    # ax[0, 1].imshow(np.asarray(reconstructed_image.squeeze().detach().cpu().numpy(), dtype=np.int8), cmap='gray')
    # ax[1, 0].imshow(np.asarray(sample1[0].squeeze().detach().cpu().numpy(), dtype=np.int8), cmap='gray')
    # ax[1, 1].imshow(np.asarray(sample1[0].squeeze().detach().cpu().numpy(), dtype=np.int8), cmap='gray')
    k = 0
    for i in range(4):
        for j in range(4):
            if i == 0 and j == 0:
                ax[0, j].imshow(np.asarray(x.squeeze().detach().cpu().numpy(), dtype=np.int8), cmap='gray')
            elif i == 0 and j == 1:
                ax[0, j].imshow(np.asarray(reconstructed_image.squeeze().detach().cpu().numpy(), dtype=np.int8), cmap='gray')

            else:
                ax[i, j].imshow(np.asarray(sample1[k].squeeze().detach().cpu().numpy(), dtype=np.int8), cmap='gray')
                k += 1
    plt.savefig("image_glow.png")
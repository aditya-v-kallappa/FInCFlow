import torch
from layers.conv import PaddedConv2d
from utils.solve_mc import solve
import time
from fastflow import FastFlowUnit

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

def test_PaddedConv2d(x=None, w=None, bias=False, order='TL', is_input=True, print_answer=False):
    if x is None:
        x = torch.randn((1, 2, 3, 3))
    B, C, H, W = x.shape
    if w is None:
        K_H, K_W = 3, 3
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
    
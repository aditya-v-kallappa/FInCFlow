import torch
import torchvision
from layers.conv import PaddedConv2d
from utils.solve_mc import solve
import time
from fastflow import FastFlowUnit
import matplotlib.pyplot as plt
import numpy as np
from layers.split import Split

from layers import Dequantization, Normalization
from layers.distributions.uniform import UniformDistribution
# from layers.splitprior import SplitPrior
from layers.activations import SigmoidFlow
from fastflow_cifar_multi_gpu import SplitPrior
from layers.flowsequential import FlowSequential
from layers.conv1x1 import Conv1x1
from layers.actnorm import ActNorm
from layers.squeeze import Squeeze
from layers.transforms import LogitTransform
from layers.coupling import Coupling
from train.losses import NegativeGaussianLoss
from train.experiment import Experiment
from datasets.mnist import load_data
from datasets.cifar10 import load_data as load_data_cifar
from layers.flowlayer import ModifiedGradFlowLayer
from fastflow_mnist import create_model as create_model_fastflow
from glow_mnist import create_model as create_model_glow
# from fastflow_cifar_multi_gpu import FastFlow, Split
from fastflow_mnist_multi_gpu import FastFlow as FastFlow_MNIST
from fastflow_mnist_multi_gpu import Split
from fastflow_cifar_multi_gpu import FastFlow as FastFlow_CIFAR
from fastflow_cifar_multi_gpu import FastFlow as FastFlow
from fastflow_imagenet64_multi_gpu import FastFlow as FastFlow_Imagenet64
from fastflow_imagenet_multi_gpu import FastFlow as FastFlow_Imagenet32
from datasets.imagenet import load_data as load_data_imagenet
from test_cifar import FastFlow as FastFlowSigmoid
from fastflow_cifar_multi_gpu import Preprocess, GlowStep, FastFlowStep, FastFlowLevel, Split, Gaussianize

from experiments.selfnorm_glow_mnist import create_model as create_model_snf_mnist
from experiments.selfnorm_glow_cifar import create_model as create_model_snf_cifar
from experiments.selfnorm_glow_imagenet import create_model as create_model_snf_imagenet
from experiments.emerging_glow import create_model as create_model_emerging
from cinc_mnist import create_model as create_model_cinc


imagenet64_data_dir = '/scratch/aditya.kallappa/Imagenet'

# from prettytable import PrettyTable

# def count_parameters(model):
#     table = PrettyTable(["Modules", "Parameters"])
#     total_params = 0
#     for name, parameter in model.named_parameters():
#         if not parameter.requires_grad: continue
#         params = parameter.numel()
#         table.add_row([name, params])
#         total_params+=params
#     # print(table)
#     # print(f"Total Trainable Params: {total_params}")
#     return total_params
    

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
    layer = SplitPrior(size=(C, H, W), distribution=NegativeGaussianLoss).to("cuda")
    if is_input:
        forward_output, z, _ = layer.forward(x)
        reverse_input = layer.reverse(forward_output, z)
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


def reconstruct(model, x, true_inverse=True):
    input = x
    zs = []
    z_inv_index = -1
    # print("Input:", input.shape)
    # Forward
    for module in model.sequence_modules:
        
        if isinstance(module, SplitPrior) and true_inverse:
            output, z, layer_logdet = module(input, true_inverse)
            zs.append(z)
        else:
            output, layer_logdet = module(input)
        # print(module, output.shape)
        input = output
    
    # Reverse
    for module in reversed(model.sequence_modules):
        if isinstance(module, SplitPrior) and true_inverse:
            input = torch.cat([input, zs[z_inv_index]], dim=1)
            output = module.reverse(input, true_inverse)
            z_inv_index -= 1
        else:
            output = module.reverse(input)
        input = output
    
    return input

def reconstruct_ff(model, x, true_inverse=True):
    input = x
    zs = []
    x, _ = model.module.preprocess(x)
    # print("After preprocess", x.shape, torch.cuda.current_device())
    for module in model.module.fastflow_levels:
        x, z, _ = module(x)
        zs.append(z)
    # print("After fastflow_levels", x.shape, torch.cuda.current_device())
    x, _ = model.module.squeeze(x)
    # print("Outside Squeeze:", x.shape, torch.cuda.current_device())
    for module in model.module.fastflow_step:
        x, _ = module(x)  
    zs.append(x)
    
    # print("Final:", x.mean())
    z = zs.pop()
    if true_inverse:
        #Reverse

        for module in reversed(model.module.fastflow_step):
            # print(module)
            z = module.reverse(z)
        
            # print("After fastflow step:", z.mean())

        x = model.module.squeeze.reverse(z)

        for m in reversed(model.module.fastflow_levels):
            # print(m)
            # z = z_std * (model.module.base_dist.sample(x.shape).squeeze() if len(zs)==1 else zs[-i-2])  # if no z's are passed, generate new random numbers from the base dist
            if zs == []:
                x = m.reverse(x)#, z)
            else:
                x = m.reverse(x, zs.pop())

        # print("After fastflow level:", x.mean())

        # postprocess
        x = model.module.preprocess.reverse(x)
        
    else:
        x = model.module.reverse(x.shape[0], [z])

    return x

def test_FastFlowMNIST(model=None, checkpoint_path=None, x=None, batch_size=25, plot=True, true_inverse=True):
    if model is None:
        model = create_model_fastflow(num_blocks=2,
                            block_size=16, 
                            sym_recon_grad=False,
                            actnorm=True,
                            split_prior=True,
                            recon_loss_weight=1.0).to('cuda')
    check = 'untrained'
    if checkpoint_path:
        print("Loading from ", checkpoint_path)

        checkpoint = torch.load(checkpoint_path)
        summary = checkpoint['summary']
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        check = 'trained'

    if not x:
        train_loader, val_loader, test_loader = load_data(data_aug=False, batch_size=batch_size)
        for x, label in test_loader:
            x = x.cuda()
            break
    else:
        x = x.cuda()
    
    model.eval()

    reconstructed_image = reconstruct(model, x, true_inverse)
    # sample1, _ = model.sample(n_samples=1, also_true_inverse=False)
    # if plot:
    #     fig, ax = plt.subplots(5, 5)
    #     # print(x)
    #     # print(reconstructed_image)
    #     # print(sample1)
    #     # print(sample2)

    #     # ax[0, 0].imshow(np.asarray(x.squeeze().detach().cpu().numpy(), dtype=np.int8), cmap='gray')
    #     # ax[0, 1].imshow(np.asarray(reconstructed_image.squeeze().detach().cpu().numpy(), dtype=np.int8), cmap='gray')
    #     # ax[1, 0].imshow(np.asarray(sample1[0].squeeze().detach().cpu().numpy(), dtype=np.int8), cmap='gray')
    #     # ax[1, 1].imshow(np.asarray(sample1[0].squeeze().detach().cpu().numpy(), dtype=np.int8), cmap='gray')
    #     k = 0
    #     m = 2
    #     for i in range(4):
    #         for j in range(4):
    #             if i == 0 and j == 0:
    #                 ax[0, j].imshow(np.asarray(x[m].squeeze().detach().cpu().numpy(), dtype=np.int8), cmap='gray')
    #             elif i == 0 and j == 1:
    #                 ax[0, j].imshow(np.asarray(reconstructed_image[m].squeeze().detach().cpu().numpy(), dtype=np.int8), cmap='gray')

    #             else:
    #                 ax[i, j].imshow(np.asarray(sample1[k].squeeze().detach().cpu().numpy(), dtype=np.int8), cmap='gray')
    #                 k += 1
    #     plt.savefig("image_fastflow.png")
    
    # else:
    #     print(sample1)
    #     forward_sample = model.forward(sample1)
        
    #     print("Error:", torch.mean((sample1 - forward_sample) ** 2).item())

    if plot:
        torchvision.utils.save_image(
            reconstructed_image / 256., f'{true_inverse}_reconstruct_{check}.png', nrow=10,
            padding=2, normalize=False)

        torchvision.utils.save_image(
            x / 256., f'{true_inverse}_original_{check}.png', nrow=10,
            padding=2, normalize=False)

    print("Error:", torch.mean((x - reconstructed_image) ** 2).item())


def test_GlowMNIST(model=None, checkpoint_path=None, x=None, batch_size=1):
    if model is None:
        model = create_model_glow(num_blocks=3,
                            block_size=32, 
                            sym_recon_grad=False,
                            actnorm=True,
                            split_prior=True,
                            current_size=(3, 32, 32),
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


def test_FastFlow(model=None, checkpoint_path=None, x=None, batch_size=25, plot=True, true_inverse=True):
    if model is None:
        model = FastFlow(n_blocks=3,
                            block_size=32,
                            actnorm=True,
                            image_size=(3, 32, 32)).to('cuda')
    check = 'untrained'
    # model = torch.nn.DataParallel(model)
    if checkpoint_path:
        model = torch.nn.DataParallel(model)
        print("Loading from ", checkpoint_path)

        checkpoint = torch.load(checkpoint_path)
        summary = checkpoint['summary']
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        check = 'trained'

    if not x:
        # train_loader, val_loader, test_loader = load_data(data_aug=False, batch_size=batch_size)
        train_loader, val_loader, test_loader = load_data_cifar(data_aug=True, batch_size=batch_size)
        for x, label in test_loader:
            x = x.cuda()
            break
    else:
        x = x.cuda()
    
    model.eval()
    with torch.no_grad():
        # x_hat = reconstruct_ms(model, x, true_inverse)
        # x_hat = model.reconstruct(x, true_inverse=true_inverse)
        x_hat = reconstruct_ff(model, x, true_inverse)

    if plot:
        torchvision.utils.save_image(
            x_hat / 256., f'test_images/{true_inverse}_cifar_reconstructFF_{check}.png', nrow=10,
            padding=2, normalize=False)

        torchvision.utils.save_image(
            x / 256., f'test_images/{true_inverse}_cifar_originalFF_{check}.png', nrow=10,
            padding=2, normalize=False)

    print("Error:", torch.mean((x - x_hat) ** 2).item())





def test_Preprocess(x, is_input=True):
    B, C, H, W = x.shape
    layer = Preprocess(size=(C, H, W))
    if is_input:
        forward_output, _ = layer.forward(x)
        reverse_input = layer.reverse(forward_output)
        print("Error:", torch.mean((x - reverse_input) ** 2).item())
    else:
        reverse_input = layer.reverse(x)
        forward_output, _ = layer.forward(reverse_input)
        print("Error:", torch.mean((x - forward_output) ** 2).item())

def test_GlowStep(x, is_input=True):
    B, C, H, W = x.shape
    layer = GlowStep(size=(C, H, W), actnorm=True)
    if is_input:
        forward_output, _ = layer.forward(x)
        reverse_input = layer.reverse(forward_output)
        print("Error:", torch.mean((x - reverse_input) ** 2).item())
    else:
        reverse_input = layer.reverse(x)
        forward_output, _ = layer.forward(reverse_input)
        print("Error:", torch.mean((x - forward_output) ** 2).item())

def test_FastFlowStep(x, is_input=True):
    B, C, H, W = x.shape
    layer = FastFlowStep(size=(C, H, W), actnorm=True)
    if is_input:
        forward_output, _ = layer.forward(x)
        reverse_input = layer.reverse(forward_output)
        print("Error:", torch.mean((x - reverse_input) ** 2).item())
    else:
        reverse_input = layer.reverse(x)
        forward_output, _ = layer.forward(reverse_input)
        print("Error:", torch.mean((x - forward_output) ** 2).item())

def test_Split(x, is_input=True):
    B, C, H, W = x.shape
    layer = Split((C, H, W), NegativeGaussianLoss)
    if is_input:
        forward_output, z2, _ = layer.forward(x)
        reverse_input = layer.reverse(forward_output, z2)
        print("Error:", torch.mean((x - reverse_input) ** 2).item())
    else:
        reverse_input = layer.reverse(x)
        forward_output, _ = layer.forward(reverse_input)
        print("Error:", torch.mean((x - forward_output) ** 2).item())

def test_Gaussianize(x, is_input=True, with_zeros=False):
    B, C, H, W = x.shape
    if not with_zeros:
        assert not C % 2, "Error: channels not divisible by 2"
        x1, x2 = x.chunk(2, dim=1)
        layer = Gaussianize(C // 2)
        
        if is_input:
            forward_output, _ = layer.forward(x1, x2)
            reverse_input = layer.reverse(x1, forward_output)
            print("Error:", torch.mean((forward_output - reverse_input) ** 2).item())
        else:
            reverse_input = layer.reverse(x)
            forward_output, _ = layer.forward(reverse_input)
            print("Error:", torch.mean((x - forward_output) ** 2).item())

    else:
        x1 = torch.zeros_like(x)
        x2 = x
        layer = Gaussianize(C)
        if is_input:
            forward_output, _ = layer.forward(x1, x2)
            reverse_input = layer.reverse(x1, forward_output)
            print("Error:", torch.mean((forward_output - reverse_input) ** 2).item())

def test_FastFlowLevel(x, is_input=True):
    B, C, H, W = x.shape
    layer = FastFlowLevel(size=(C, H, W), block_size=16, actnorm=True)
    if is_input:
        forward_output, z2, _ = layer.forward(x)
        reverse_input = layer.reverse(forward_output, z2)
        print("Error:", torch.mean((x - reverse_input) ** 2).item())
    else:
        reverse_input = layer.reverse(x)
        forward_output, _ = layer.forward(reverse_input)
        print("Error:", torch.mean((x - forward_output) ** 2).item())

def test_FastFlow_(x, is_input=True):
    B, C, H, W = x.shape
    layer = FastFlow(n_blocks=3, block_size=16, image_size=(C, H, W), actnorm=True)
    if is_input:
        forward_output, _ = layer.forward(x)
        reverse_input = layer.reverse(n_samples=B, zs=forward_output)
        print("Error:", torch.mean((x - reverse_input) ** 2).item())
    else:
        reverse_input = layer.reverse(x)
        forward_output, _ = layer.forward(reverse_input)
        print("Error:", torch.mean((x - forward_output) ** 2).item())

def test_inverse_PaddedConv2d(in_channels=3, out_channels=None, kernel_size=(3, 3), bias=True, order='TL', batch_size=1, image_size=(3, 3)):
    print("-----------------------------------------------------")
    if out_channels is None:
        out_channels = in_channels
    assert in_channels == out_channels, "Input and Output channels have to be the same"
    padded_conv = PaddedConv2d(in_channels, out_channels, kernel_size, bias, order).cuda()
    padded_conv.eval()
    # print("Input : \n", np.array(input))
    # print("Kernel : \n", np.array(kernel))
    input = torch.randn((batch_size, in_channels, image_size[0], image_size[1]))
    input = torch.tensor(input, dtype=torch.float).cuda()
    print("Input Shape:", input.shape)
    print("Order:", order)
    print("Kernel: ", kernel_size)
    B, C, H, W = input.shape
    output = torch.zeros((B, C, H, W), dtype=torch.float).cuda()
    t = time.process_time()
    conv_output, _ = padded_conv(input)
    forward_time = time.process_time() - t


    t = time.process_time()
    reverse_input_cython, _ = padded_conv.reverse(conv_output)
    reverse_time_cython = time.process_time() - t
    
    error_cython = torch.mean((input - reverse_input_cython) ** 2)

    t = time.process_time()
    # reverse_input_python, _ = padded_conv.reverse_python(conv_output)
    reverse_time_python = time.process_time() - t
    
    # error_python = torch.mean((input - reverse_input_python) ** 2)


    t = time.process_time()
    reverse_input_cuda, _ = padded_conv.reverse_cuda(conv_output)
    reverse_time_cuda = time.process_time() - t
    
    error_cuda = torch.mean((input - reverse_input_cuda) ** 2)

    # print(f"MSE Python : {error_python}")
    print(f"MSE Cython : {error_cython}")
    print(f"MSE Cuda Level 1 : {error_cuda}")

    print(f"Forward Time : {forward_time}s")
    # print(f"Reverse Time Python: {reverse_time_python}s")
    print(f"Reverse Time Cython: {reverse_time_cython}s")
    print(f"Reverse Time Cuda Level 1: {reverse_time_cuda}s")


def test_inverse_FastFlowUnit(in_channels=4, out_channels=None, kernel_size=(3, 3), batch_size=1, image_size=(3, 3)):
    print("-----------------------------------------------------")
    if out_channels is None:
        out_channels = in_channels
    assert in_channels == out_channels, "Input and Output channels have to be the same"
    fastflowunit = FastFlowUnit(in_channels, out_channels, kernel_size).cuda()
    # print("Input : \n", np.array(input))
    # print("Kernel : \n", np.array(kernel))
    input = torch.randn((batch_size, in_channels, image_size[0], image_size[1]))
    input = torch.tensor(input, dtype=torch.float).cuda()
    
    B, C, H, W = input.shape
    output = torch.zeros((B, C, H, W), dtype=torch.float).cuda()
    t = time.process_time()
    conv_output, _ = fastflowunit(input)
    forward_time = time.process_time() - t
    t = time.process_time()
    reverse_input = fastflowunit.reverse(conv_output)
    reverse_time = time.process_time() - t

    t = time.process_time()
    reverse_input_level2 = fastflowunit.reverse_level2(conv_output)
    reverse_time_level2 = time.process_time() - t
    
    error_level1 = torch.mean((input - reverse_input) ** 2)
    error_level2 = torch.mean((input - reverse_input_level2) ** 2)

    print(f"MSE Level 1 : {error_level1}")
    print(f"MSE Level 2 : {error_level2}")
    print(f"Forward Time : {forward_time}s")
    print(f"Level 1 Reverse Time : {reverse_time}s")
    print(f"Level 2 Reverse Time : {reverse_time_level2}s")



def _test_inverse_FastFlow_MNIST(n_blocks=2, block_size= 16, batch_size=100, image_size=(1, 28, 28)):

    fastflow = FastFlow_MNIST(n_blocks=n_blocks,
                        block_size=block_size,
                        actnorm=True,
                        image_size=image_size).to('cuda')
    print("-----------------------------------------------------")
    in_channels = image_size[0]
    out_channels = in_channels

    total_params_learnable = sum(p.numel() for p in fastflow.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in fastflow.parameters())
    # print("Input : \n", np.array(input))
    # print("Kernel : \n", np.array(kernel))
    # input = torch.randn((batch_size, in_channels, image_size[1], image_size[2]))
    # input = torch.tensor(input, dtype=torch.float).cuda()
    train_loader, val_loader, test_loader = load_data(data_aug=False, batch_size=batch_size)
    for x, label in test_loader:
        input = x.cuda()
        break
    B, C, H, W = input.shape
    # output = torch.zeros((B, C, H, W), dtype=torch.float).cuda()
    total_time_forward = 0
    total_time_reverse = 0
    n_loops = 10
    with torch.no_grad():
        for i in range(n_loops + 1):    
            t = time.process_time()
            conv_output, _ = fastflow(input)
            forward_time = time.process_time() - t

            t = time.process_time()
            reverse_input, _ = fastflow.sample(n_samples=batch_size)
            reverse_time = time.process_time() - t
            if i != 0:
                total_time_forward += forward_time
                total_time_reverse += reverse_time

    # t = time.process_time()
    # reverse_input_level2 = fastflow.reverse_level2(conv_output)
    # reverse_time_level2 = time.process_time() - t
    
    # error_level1 = torch.mean((input - reverse_input) ** 2)
    # error_level2 = torch.mean((input - reverse_input_level2) ** 2)

    # print(f"MSE Level 1 : {error_level1}")
    # print(f"MSE Level 2 : {error_level2}")
    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    # for param in fastflow.state_dict():
    #     print(param, torch.numel(fastflow.state_dict()[param]))
    # for p in fastflow.parameters():
    #     print(p.shape)
    print(f"Average Forward Time : {total_time_forward / n_loops}s")
    print(f"Average Level 1 Reverse Time : {total_time_reverse / n_loops}s")
    # print(f"Level 2 Reverse Time : {reverse_time_level2}s")

def test_inverse_FastFlow_CIFAR(n_blocks=3, block_size=32, batch_size=100, image_size=(3, 32, 32)):

    fastflow = FastFlow_CIFAR(n_blocks=n_blocks,
                        block_size=block_size,
                        actnorm=True,
                        image_size=image_size).to('cuda')
    fastflow.eval()
    print("-----------------------------------------------------")
    in_channels = image_size[0]
    out_channels = in_channels

    total_params_learnable = sum(p.numel() for p in fastflow.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in fastflow.parameters())
    # print("Input : \n", np.array(input))
    # print("Kernel : \n", np.array(kernel))
    # input = torch.randn((batch_size, in_channels, image_size[1], image_size[2]))
    # input = torch.tensor(input, dtype=torch.float).cuda()
    train_loader, val_loader, test_loader = load_data_cifar(data_aug=True, batch_size=batch_size)
    for x, label in test_loader:
        input = x.cuda()
        break
    print("Input", x.shape)
    B, C, H, W = input.shape
    # output = torch.zeros((B, C, H, W), dtype=torch.float).cuda()
    total_time_forward = 0
    total_time_reverse = 0
    n_loops = 10
    with torch.no_grad():
        for i in range(n_loops + 1):    
            t = time.process_time()
            conv_output, _ = fastflow(input)
            forward_time = time.process_time() - t

            t = time.process_time()
            reverse_input, _ = fastflow.sample(n_samples=batch_size)
            reverse_time = time.process_time() - t
            if i != 0:
                total_time_forward += forward_time
                total_time_reverse += reverse_time

    # t = time.process_time()
    # reverse_input_level2 = fastflow.reverse_level2(conv_output)
    # reverse_time_level2 = time.process_time() - t
    
    # error_level1 = torch.mean((input - reverse_input) ** 2)
    # error_level2 = torch.mean((input - reverse_input_level2) ** 2)

    # print(f"MSE Level 1 : {error_level1}")
    # print(f"MSE Level 2 : {error_level2}")
    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    # for param in fastflow.state_dict():
    #     print(param, torch.numel(fastflow.state_dict()[param]))
    # for p in fastflow.parameters():
    #     print(p.shape)
    print(f"Average Forward Time : {total_time_forward / n_loops}s")
    print(f"Average Level 1 Reverse Time : {total_time_reverse / n_loops}s")
    # print(f"Level 2 Reverse Time : {reverse_time_level2}s")

def test_inverse_FastFlow_MNIST(n_blocks=2, block_size= 16, batch_size=100, image_size=(1, 28, 28)):

    fastflow = FastFlow_MNIST(n_blocks=n_blocks,
                        block_size=block_size,
                        actnorm=True,
                        image_size=image_size).to('cuda')
    print("-----------------------------------------------------")
    in_channels = image_size[0]
    out_channels = in_channels

    total_params_learnable = sum(p.numel() for p in fastflow.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in fastflow.parameters())
    # print("Input : \n", np.array(input))
    # print("Kernel : \n", np.array(kernel))
    # input = torch.randn((batch_size, in_channels, image_size[1], image_size[2]))
    # input = torch.tensor(input, dtype=torch.float).cuda()
    train_loader, val_loader, test_loader = load_data(data_aug=False, batch_size=batch_size)
    for x, label in test_loader:
        input = x.cuda()
        break
    B, C, H, W = input.shape
    # output = torch.zeros((B, C, H, W), dtype=torch.float).cuda()
    total_time_forward = 0
    total_time_reverse = 0
    n_loops = 10
    with torch.no_grad():
        for i in range(n_loops + 1):    
            t = time.process_time()
            conv_output, _ = fastflow(input)
            forward_time = time.process_time() - t

            t = time.process_time()
            reverse_input, _ = fastflow.sample(n_samples=batch_size)
            reverse_time = time.process_time() - t
            if i != 0:
                total_time_forward += forward_time
                total_time_reverse += reverse_time

    # t = time.process_time()
    # reverse_input_level2 = fastflow.reverse_level2(conv_output)
    # reverse_time_level2 = time.process_time() - t
    
    # error_level1 = torch.mean((input - reverse_input) ** 2)
    # error_level2 = torch.mean((input - reverse_input_level2) ** 2)

    # print(f"MSE Level 1 : {error_level1}")
    # print(f"MSE Level 2 : {error_level2}")
    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    # for param in fastflow.state_dict():
    #     print(param, torch.numel(fastflow.state_dict()[param]))
    # for p in fastflow.parameters():
    #     print(p.shape)
    print(f"Average Forward Time : {total_time_forward / n_loops}s")
    print(f"Average Level 1 Reverse Time : {total_time_reverse / n_loops}s")
    # print(f"Level 2 Reverse Time : {reverse_time_level2}s")

def test_inverse_FastFlow_CIFAR2(n_blocks=3, block_size=32, batch_size=100, image_size=(3, 32, 32)):

    fastflow = create_model_fastflow(num_blocks=n_blocks,
                        block_size=block_size,
                        actnorm=True,
                        split_prior=True,
                        current_size=image_size).to('cuda')
    fastflow.eval()
    print("-----------------------------------------------------")
    in_channels = image_size[0]
    out_channels = in_channels

    total_params_learnable = sum(p.numel() for p in fastflow.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in fastflow.parameters())
    # print("Input : \n", np.array(input))
    # print("Kernel : \n", np.array(kernel))
    # input = torch.randn((batch_size, in_channels, image_size[1], image_size[2]))
    # input = torch.tensor(input, dtype=torch.float).cuda()
    train_loader, val_loader, test_loader = load_data_cifar(data_aug=True, batch_size=batch_size)
    for x, label in test_loader:
        input = x.cuda()
        break
    print("Input", x.shape)
    B, C, H, W = input.shape
    # output = torch.zeros((B, C, H, W), dtype=torch.float).cuda()
    total_time_forward = 0
    total_time_reverse = 0
    n_loops = 10
    with torch.no_grad():
        for i in range(n_loops + 1):    
            t = time.process_time()
            conv_output, _ = fastflow(input)
            forward_time = time.process_time() - t

            t = time.process_time()
            reverse_input, _ = fastflow.sample(n_samples=batch_size)
            reverse_time = time.process_time() - t
            if i != 0:
                total_time_forward += forward_time
                total_time_reverse += reverse_time

    # t = time.process_time()
    # reverse_input_level2 = fastflow.reverse_level2(conv_output)
    # reverse_time_level2 = time.process_time() - t
    
    # error_level1 = torch.mean((input - reverse_input) ** 2)
    # error_level2 = torch.mean((input - reverse_input_level2) ** 2)

    # print(f"MSE Level 1 : {error_level1}")
    # print(f"MSE Level 2 : {error_level2}")
    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    # for param in fastflow.state_dict():
    #     print(param, torch.numel(fastflow.state_dict()[param]))
    # for p in fastflow.parameters():
    #     print(p.shape)
    print(f"Average Forward Time : {total_time_forward / n_loops}s")
    print(f"Average Level 1 Reverse Time : {total_time_reverse / n_loops}s")
    # print(f"Level 2 Reverse Time : {reverse_time_level2}s")


def test_inverse_FastFlow_Imagenet64(n_blocks=3, block_size=32, batch_size=100, image_size=(3, 32, 32)):

    fastflow = FastFlow_Imagenet64(n_blocks=n_blocks,
                        block_size=block_size,
                        actnorm=True,
                        image_size=image_size).to('cuda')
    fastflow.eval()
    print("-----------------------------------------------------")
    in_channels = image_size[0]
    out_channels = in_channels

    total_params_learnable = sum(p.numel() for p in fastflow.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in fastflow.parameters())
    # print("Input : \n", np.array(input))
    # print("Kernel : \n", np.array(kernel))
    # input = torch.randn((batch_size, in_channels, image_size[1], image_size[2]))
    # input = torch.tensor(input, dtype=torch.float).cuda()
    # train_loader, val_loader, test_loader = load_data_imagenet(data_aug=True, batch_size=batch_size)
    train_loader, val_loader, test_loader = load_data_imagenet(data_aug=True, 
                                                      batch_size=batch_size,
                                                      resolution=64,
                                                      data_dir=imagenet64_data_dir)
    for x, label in test_loader:
        input = x.cuda()
        break
    print("Input", input.shape)
    B, C, H, W = input.shape
    # output = torch.zeros((B, C, H, W), dtype=torch.float).cuda()
    total_time_forward = 0
    total_time_reverse = 0
    n_loops = 10
    with torch.no_grad():
        for i in range(n_loops + 1):    
            t = time.process_time()
            conv_output, _ = fastflow(input)
            forward_time = time.process_time() - t

            t = time.process_time()
            reverse_input, _ = fastflow.sample(n_samples=batch_size)
            reverse_time = time.process_time() - t
            if i != 0:
                total_time_forward += forward_time
                total_time_reverse += reverse_time


    # t = time.process_time()
    # reverse_input_level2 = fastflow.reverse_level2(conv_output)
    # reverse_time_level2 = time.process_time() - t
    
    # error_level1 = torch.mean((input - reverse_input) ** 2)
    # error_level2 = torch.mean((input - reverse_input_level2) ** 2)

    # print(f"MSE Level 1 : {error_level1}")
    # print(f"MSE Level 2 : {error_level2}")
    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    print(f"Forward Time : {total_time_forward / n_loops}s")
    print(f"Level 2 Reverse Time : {total_time_reverse / n_loops}s")
    # print(f"Level 2 Reverse Time : {reverse_time_level2}s")


def test_inverse_FastFlow_Imagenet32(n_blocks=3, block_size=48, batch_size=100, image_size=(3, 32, 32)):

    fastflow = FastFlow_Imagenet32(n_blocks=n_blocks,
                        block_size=block_size,
                        actnorm=True,
                        image_size=image_size).to('cuda')
    fastflow.eval()
    print("-----------------------------------------------------")
    in_channels = image_size[0]
    out_channels = in_channels

    total_params_learnable = sum(p.numel() for p in fastflow.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in fastflow.parameters())
    # print("Input : \n", np.array(input))
    # print("Kernel : \n", np.array(kernel))
    # input = torch.randn((batch_size, in_channels, image_size[1], image_size[2]))
    # input = torch.tensor(input, dtype=torch.float).cuda()
    # train_loader, val_loader, test_loader = load_data_imagenet(data_aug=True, batch_size=batch_size)
    train_loader, val_loader, test_loader = load_data_imagenet(data_aug=True, 
                                                      batch_size=batch_size,
                                                      resolution=32,
                                                      data_dir=imagenet64_data_dir)
    for x, label in test_loader:
        input = x.cuda()
        break
    print("Input", input.shape)
    B, C, H, W = input.shape
    # output = torch.zeros((B, C, H, W), dtype=torch.float).cuda()
    total_time_forward = 0
    total_time_reverse = 0
    n_loops = 10
    with torch.no_grad():
        for i in range(n_loops + 1):    
            t = time.process_time()
            conv_output, _ = fastflow(input)
            forward_time = time.process_time() - t

            t = time.process_time()
            reverse_input, _ = fastflow.sample(n_samples=batch_size)
            reverse_time = time.process_time() - t
            if i != 0:
                total_time_forward += forward_time
                total_time_reverse += reverse_time


    # t = time.process_time()
    # reverse_input_level2 = fastflow.reverse_level2(conv_output)
    # reverse_time_level2 = time.process_time() - t
    
    # error_level1 = torch.mean((input - reverse_input) ** 2)
    # error_level2 = torch.mean((input - reverse_input_level2) ** 2)

    # print(f"MSE Level 1 : {error_level1}")
    # print(f"MSE Level 2 : {error_level2}")
    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    print(f"Forward Time : {total_time_forward / n_loops}s")
    print(f"Level 2 Reverse Time : {total_time_reverse / n_loops}s")
    # print(f"Level 2 Reverse Time : {reverse_time_level2}s")



def test_inverse_Glow_MNIST(n_blocks=2, block_size= 16, batch_size=100, image_size=(1, 28, 28)):

    glow = create_model_glow(num_blocks=n_blocks,
                        block_size=block_size,
                        actnorm=True,
                        current_size=image_size,
                        split_prior=True).to('cuda')
    print("-----------------------------------------------------")
    in_channels = image_size[0]
    out_channels = in_channels

    total_params_learnable = sum(p.numel() for p in glow.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in glow.parameters())
    train_loader, val_loader, test_loader = load_data(data_aug=False, batch_size=batch_size)
    for x, label in test_loader:
        input = x.cuda()
        break
    B, C, H, W = input.shape
    total_time_forward = 0
    total_time_reverse = 0
    n_loops = 10
    with torch.no_grad():
        for i in range(n_loops + 1):    
            t = time.process_time()
            conv_output, _ = glow(input)
            forward_time = time.process_time() - t

            t = time.process_time()
            reverse_input, _ = glow.sample(n_samples=batch_size)
            reverse_time = time.process_time() - t
            if i != 0:
                total_time_forward += forward_time
                total_time_reverse += reverse_time

    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    
    print(f"Average Forward Time : {total_time_forward / n_loops}s")
    print(f"Average Level 1 Reverse Time : {total_time_reverse / n_loops}s")
    # print(f"Level 2 Reverse Time : {reverse_time_level2}s")



def test_inverse_Glow_CIFAR(n_blocks=3, block_size=32, batch_size=100, image_size=(3, 32, 32)):

    glow = create_model_glow(num_blocks=n_blocks,
                        block_size=block_size,
                        actnorm=True,
                        current_size=image_size,
                        split_prior=True).to('cuda')
    print("-----------------------------------------------------")
    in_channels = image_size[0]
    out_channels = in_channels

    total_params_learnable = sum(p.numel() for p in glow.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in glow.parameters())

    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    # for param in glow.state_dict(): 
    #     print(param, torch.numel(glow.state_dict()[param]))
    # for p in glow.parameters():
    #     print(p.shape)
    # print("Input : \n", np.array(input))
    # print("Kernel : \n", np.array(kernel))
    # input = torch.randn((batch_size, in_channels, image_size[1], image_size[2]))
    # input = torch.tensor(input, dtype=torch.float).cuda()
    train_loader, val_loader, test_loader = load_data_cifar(data_aug=True, batch_size=batch_size)
    for x, label in test_loader:
        input = x.cuda()
        break
    B, C, H, W = input.shape
    # output = torch.zeros((B, C, H, W), dtype=torch.float).cuda()
    total_time_forward = 0
    total_time_reverse = 0
    n_loops = 10
    with torch.no_grad():
        for i in range(n_loops + 1):    
            t = time.process_time()
            conv_output, _ = glow(input)
            forward_time = time.process_time() - t

            t = time.process_time()
            reverse_input, _ = glow.sample(n_samples=batch_size)
            reverse_time = time.process_time() - t
            if i != 0:
                total_time_forward += forward_time
                total_time_reverse += reverse_time

    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    
    print(f"Average Forward Time : {total_time_forward / n_loops}s")
    print(f"Average Level 1 Reverse Time : {total_time_reverse / n_loops}s")


def test_inverse_Glow_Imagenet32(n_blocks=3, block_size=32, batch_size=100, image_size=(3, 32, 32)):

    glow = create_model_glow(num_blocks=n_blocks,
                        block_size=block_size,
                        actnorm=True,
                        current_size=image_size,
                        split_prior=True).to('cuda')
    print("-----------------------------------------------------")
    in_channels = image_size[0]
    out_channels = in_channels

    total_params_learnable = sum(p.numel() for p in glow.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in glow.parameters())

    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    # for param in glow.state_dict(): 
    #     print(param, torch.numel(glow.state_dict()[param]))
    # for p in glow.parameters():
    #     print(p.shape)
    # print("Input : \n", np.array(input))
    # print("Kernel : \n", np.array(kernel))
    # input = torch.randn((batch_size, in_channels, image_size[1], image_size[2]))
    # input = torch.tensor(input, dtype=torch.float).cuda()
    train_loader, val_loader, test_loader = load_data_imagenet(data_aug=True, 
                                                      batch_size=batch_size,
                                                      resolution=32,
                                                      data_dir=imagenet64_data_dir)
    for x, label in test_loader:
        input = x.cuda()
        break
    B, C, H, W = input.shape
    # output = torch.zeros((B, C, H, W), dtype=torch.float).cuda()
    total_time_forward = 0
    total_time_reverse = 0
    n_loops = 10
    with torch.no_grad():
        for i in range(n_loops + 1):    
            t = time.process_time()
            conv_output, _ = glow(input)
            forward_time = time.process_time() - t

            t = time.process_time()
            reverse_input, _ = glow.sample(n_samples=batch_size)
            reverse_time = time.process_time() - t
            if i != 0:
                total_time_forward += forward_time
                total_time_reverse += reverse_time

    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    
    print(f"Average Forward Time : {total_time_forward / n_loops}s")
    print(f"Average Level 1 Reverse Time : {total_time_reverse / n_loops}s")


def test_inverse_Glow_Imagenet64(n_blocks=3, block_size=32, batch_size=100, image_size=(3, 64, 64)):

    glow = create_model_glow(num_blocks=n_blocks,
                        block_size=block_size,
                        actnorm=True,
                        current_size=image_size,
                        split_prior=True).to('cuda')
    print("-----------------------------------------------------")
    in_channels = image_size[0]
    out_channels = in_channels

    total_params_learnable = sum(p.numel() for p in glow.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in glow.parameters())

    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    # for param in glow.state_dict(): 
    #     print(param, torch.numel(glow.state_dict()[param]))
    # for p in glow.parameters():
    #     print(p.shape)
    # print("Input : \n", np.array(input))
    # print("Kernel : \n", np.array(kernel))
    # input = torch.randn((batch_size, in_channels, image_size[1], image_size[2]))
    # input = torch.tensor(input, dtype=torch.float).cuda()
    train_loader, val_loader, test_loader = load_data_imagenet(data_aug=True, 
                                                      batch_size=batch_size,
                                                      resolution=64,
                                                      data_dir=imagenet64_data_dir)
    for x, label in test_loader:
        input = x.cuda()
        break
    B, C, H, W = input.shape
    # output = torch.zeros((B, C, H, W), dtype=torch.float).cuda()
    total_time_forward = 0
    total_time_reverse = 0
    n_loops = 10
    with torch.no_grad():
        for i in range(n_loops + 1):    
            t = time.process_time()
            conv_output, _ = glow(input)
            forward_time = time.process_time() - t

            t = time.process_time()
            reverse_input, _ = glow.sample(n_samples=batch_size)
            reverse_time = time.process_time() - t
            if i != 0:
                total_time_forward += forward_time
                total_time_reverse += reverse_time

    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    
    print(f"Average Forward Time : {total_time_forward / n_loops}s")
    print(f"Average Level 1 Reverse Time : {total_time_reverse / n_loops}s")



def test_inverse_FastFlowSigmoid_CIFAR(n_blocks=3, block_size=32, batch_size=100, image_size=(3, 32, 32)):

    fastflow = FastFlowSigmoid(n_blocks=n_blocks,
                        block_size=block_size,
                        actnorm=True,
                        image_size=image_size).to('cuda')
    fastflow.eval()
    print("-----------------------------------------------------")
    in_channels = image_size[0]
    out_channels = in_channels

    total_params_learnable = sum(p.numel() for p in fastflow.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in fastflow.parameters())
    # print("Input : \n", np.array(input))
    # print("Kernel : \n", np.array(kernel))
    # input = torch.randn((batch_size, in_channels, image_size[1], image_size[2]))
    # input = torch.tensor(input, dtype=torch.float).cuda()
    train_loader, val_loader, test_loader = load_data_cifar(data_aug=True, batch_size=batch_size)
    for x, label in test_loader:
        input = x.cuda()
        break
    print("Input", x.shape)
    B, C, H, W = input.shape
    # output = torch.zeros((B, C, H, W), dtype=torch.float).cuda()
    total_time_forward = 0
    total_time_reverse = 0
    n_loops = 10
    with torch.no_grad():
        for i in range(n_loops + 1):    
            t = time.process_time()
            conv_output, _ = fastflow(input)
            forward_time = time.process_time() - t

            t = time.process_time()
            reverse_input, _ = fastflow.sample(n_samples=batch_size)
            reverse_time = time.process_time() - t
            if i != 0:
                total_time_forward += forward_time
                total_time_reverse += reverse_time

    # t = time.process_time()
    # reverse_input_level2 = fastflow.reverse_level2(conv_output)
    # reverse_time_level2 = time.process_time() - t
    
    # error_level1 = torch.mean((input - reverse_input) ** 2)
    # error_level2 = torch.mean((input - reverse_input_level2) ** 2)

    # print(f"MSE Level 1 : {error_level1}")
    # print(f"MSE Level 2 : {error_level2}")
    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    # for param in fastflow.state_dict():
    #     print(param, torch.numel(fastflow.state_dict()[param]))
    # for p in fastflow.parameters():
    #     print(p.shape)
    print(f"Average Forward Time : {total_time_forward / n_loops}s")
    print(f"Average Level 1 Reverse Time : {total_time_reverse / n_loops}s")
    # print(f"Level 2 Reverse Time : {reverse_time_level2}s")


def reconstruct_sigmoid(model, x, true_inverse=True):
    input = x
    zs = []
    # z_inv_index = -1
    # # print("Input:", input.shape)
    # # Forward
    # zs, _ = model(x)
    # # print(model)
    # if true_inverse:
    #     x = model.reverse(zs=zs)
    # else:
    #     x = model.sample(n_samples=x.shape[0])
    
    # print(x.shape)

    x, _ = model.preprocess(x)
    # print("After preprocess", x.shape, torch.cuda.current_device())
    for module in model.fastflow_levels:
        x, z, _ = module(x)
        zs.append(z)
    # print("After fastflow_levels", x.shape, torch.cuda.current_device())
    x, _ = model.squeeze(x)
    # print("Outside Squeeze:", x.shape, torch.cuda.current_device())
    for module in model.fastflow_step:
        x, _ = module(x)  
    zs.append(x)
    
    print("Final:", x.mean())
    #Reverse
    z = zs.pop()
    for module in reversed(model.fastflow_step):
        print(module)
        z = module.reverse(z)
    
        print("After fastflow step:", z.mean())

    x = model.squeeze.reverse(z)

    for m in reversed(model.fastflow_levels):
        # print(m)
        # z = z_std * (model.base_dist.sample(x.shape).squeeze() if len(zs)==1 else zs[-i-2])  # if no z's are passed, generate new random numbers from the base dist
        if zs == []:
            x = m.reverse(x)#, z)
        else:
            x = m.reverse(x, zs.pop())

    print("After fastflow level:", x.mean())

    # postprocess
    x = model.preprocess.reverse(x)

    
    return x

def test_FastFlowSigmoid(n_blocks=3, block_size=32, batch_size=100, image_size=(3, 32, 32), plot=True, true_inverse=True):
    
    fastflow = FastFlowSigmoid(n_blocks=n_blocks,
                        block_size=block_size, 
                        actnorm=True,
                        image_size=image_size).to('cuda')
    check = 'untrained'
    train_loader, val_loader, test_loader = load_data_cifar(data_aug=True, batch_size=batch_size)
    i = 0
    for x, label in test_loader:
        i += 1
        x = x.cuda()
        if i == 100:
            break
       
    fastflow.eval()
    with torch.no_grad():
        reconstructed_image = reconstruct_sigmoid(fastflow, x, true_inverse)
    if plot:
        torchvision.utils.save_image(
            reconstructed_image / 256., f'test_images/{true_inverse}_Sigmoid_reconstruct_{check}.png', nrow=int(batch_size ** 0.5),
            padding=2, normalize=False)

        torchvision.utils.save_image(
            x / 256., f'test_images/{true_inverse}_Sigmoid_original_{check}.png', nrow=int(batch_size ** 0.5),
            padding=2, normalize=False)

    print("Error:", torch.mean((x - reconstructed_image) ** 2).item())

def test_SigmoidFlow(x=None):
    layer = SigmoidFlow()
    forward, _ = layer(x)
    reverse_x, _ = layer.reverse(forward)
    print("Error:", torch.mean((x - reverse_x) ** 2).item())



def test_inverse_SNF_MNIST(n_blocks=2, block_size=16, batch_size=100, image_size=(1, 28, 28)):

    glow = create_model_snf_mnist(num_blocks=n_blocks,
                        block_size=block_size,
                        actnorm=True,
                        split_prior=True).to('cuda')
    print("-----------------------------------------------------")
    in_channels = image_size[0]
    out_channels = in_channels

    total_params_learnable = sum(p.numel() for p in glow.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in glow.parameters())
    train_loader, val_loader, test_loader = load_data(data_aug=False, batch_size=batch_size)
    for x, label in test_loader:
        input = x.cuda()
        break
    B, C, H, W = input.shape
    total_time_forward = 0
    total_time_reverse = 0
    n_loops = 10
    with torch.no_grad():
        for i in range(n_loops + 1):    
            t = time.process_time()
            conv_output, _ = glow(input)
            forward_time = time.process_time() - t

            t = time.process_time()
            reverse_input, _ = glow.sample(n_samples=batch_size)
            reverse_time = time.process_time() - t
            if i != 0:
                total_time_forward += forward_time
                total_time_reverse += reverse_time

    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    
    print(f"Average Forward Time : {total_time_forward / n_loops}s")
    print(f"Average Level 1 Reverse Time : {total_time_reverse / n_loops}s")
    # print(f"Level 2 Reverse Time : {reverse_time_level2}s")



def test_inverse_SNF_CIFAR(n_blocks=3, block_size=32, batch_size=100, image_size=(3, 32, 32)):

    glow = create_model_snf_cifar(num_blocks=n_blocks,
                        block_size=block_size,
                        actnorm=True,
                        split_prior=True).to('cuda')
    print("-----------------------------------------------------")
    in_channels = image_size[0]
    out_channels = in_channels

    total_params_learnable = sum(p.numel() for p in glow.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in glow.parameters())

    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    
    train_loader, val_loader, test_loader = load_data_cifar(data_aug=True, batch_size=batch_size)
    for x, label in test_loader:
        input = x.cuda()
        break
    B, C, H, W = input.shape
    # output = torch.zeros((B, C, H, W), dtype=torch.float).cuda()
    total_time_forward = 0
    total_time_reverse = 0
    n_loops = 10
    with torch.no_grad():
        for i in range(n_loops + 1):    
            t = time.process_time()
            conv_output, _ = glow(input)
            forward_time = time.process_time() - t

            t = time.process_time()
            reverse_input, _ = glow.sample(n_samples=batch_size)
            reverse_time = time.process_time() - t
            if i != 0:
                total_time_forward += forward_time
                total_time_reverse += reverse_time

    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    
    print(f"Average Forward Time : {total_time_forward / n_loops}s")
    print(f"Average Level 1 Reverse Time : {total_time_reverse / n_loops}s")


def test_inverse_SNF_Imagenet32(n_blocks=3, block_size=48, batch_size=100, image_size=(3, 32, 32)):

    glow = create_model_snf_imagenet(num_blocks=n_blocks,
                        block_size=block_size,
                        actnorm=True,
                        split_prior=True).to('cuda')
    print("-----------------------------------------------------")
    in_channels = image_size[0]
    out_channels = in_channels

    total_params_learnable = sum(p.numel() for p in glow.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in glow.parameters())

    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    # for param in glow.state_dict(): 
    #     print(param, torch.numel(glow.state_dict()[param]))
    # for p in glow.parameters():
    #     print(p.shape)
    # print("Input : \n", np.array(input))
    # print("Kernel : \n", np.array(kernel))
    # input = torch.randn((batch_size, in_channels, image_size[1], image_size[2]))
    # input = torch.tensor(input, dtype=torch.float).cuda()
    train_loader, val_loader, test_loader = load_data_imagenet(data_aug=True, 
                                                      batch_size=batch_size,
                                                      resolution=32,
                                                      data_dir=imagenet64_data_dir)
    for x, label in test_loader:
        input = x.cuda()
        break
    B, C, H, W = input.shape
    # output = torch.zeros((B, C, H, W), dtype=torch.float).cuda()
    total_time_forward = 0
    total_time_reverse = 0
    n_loops = 10
    with torch.no_grad():
        for i in range(n_loops + 1):    
            t = time.process_time()
            conv_output, _ = glow(input)
            forward_time = time.process_time() - t

            t = time.process_time()
            reverse_input, _ = glow.sample(n_samples=batch_size)
            reverse_time = time.process_time() - t
            if i != 0:
                total_time_forward += forward_time
                total_time_reverse += reverse_time

    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    
    print(f"Average Forward Time : {total_time_forward / n_loops}s")
    print(f"Average Level 1 Reverse Time : {total_time_reverse / n_loops}s")

def test_inverse_SNF_Imagenet64(n_blocks=4, block_size=48, batch_size=100, image_size=(3, 64, 64)):

    glow = create_model_snf_imagenet(num_blocks=n_blocks,
                        block_size=block_size,
                        actnorm=True,
                        split_prior=True).to('cuda')
    print("-----------------------------------------------------")
    in_channels = image_size[0]
    out_channels = in_channels

    total_params_learnable = sum(p.numel() for p in glow.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in glow.parameters())

    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    # for param in glow.state_dict(): 
    #     print(param, torch.numel(glow.state_dict()[param]))
    # for p in glow.parameters():
    #     print(p.shape)
    # print("Input : \n", np.array(input))
    # print("Kernel : \n", np.array(kernel))
    # input = torch.randn((batch_size, in_channels, image_size[1], image_size[2]))
    # input = torch.tensor(input, dtype=torch.float).cuda()
    train_loader, val_loader, test_loader = load_data_imagenet(data_aug=True, 
                                                      batch_size=batch_size,
                                                      resolution=64,
                                                      data_dir=imagenet64_data_dir)
    for x, label in test_loader:
        input = x.cuda()
        break
    B, C, H, W = input.shape
    # output = torch.zeros((B, C, H, W), dtype=torch.float).cuda()
    total_time_forward = 0
    total_time_reverse = 0
    n_loops = 10
    with torch.no_grad():
        for i in range(n_loops + 1):    
            t = time.process_time()
            conv_output, _ = glow(input)
            forward_time = time.process_time() - t

            t = time.process_time()
            reverse_input, _ = glow.sample(n_samples=batch_size)
            reverse_time = time.process_time() - t
            if i != 0:
                total_time_forward += forward_time
                total_time_reverse += reverse_time

    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    
    print(f"Average Forward Time : {total_time_forward / n_loops}s")
    print(f"Average Level 1 Reverse Time : {total_time_reverse / n_loops}s")

def test_timings(n_blocks=3, block_size=32, batch_size=100, image_size=(3, 32, 32)):

    # fastflow = FastFlow_CIFAR(n_blocks=n_blocks,
    #                     block_size=block_size,
    #                     actnorm=True,
    #                     image_size=image_size).to('cuda')
    fastflow = create_model_fastflow(num_blocks=n_blocks,
                        block_size=block_size,
                        actnorm=True,
                        split_prior=True,
                        current_size=image_size).to('cuda')
    fastflow.eval()
    print("-----------------------------------------------------")
    in_channels = image_size[0]
    out_channels = in_channels

    total_params_learnable = sum(p.numel() for p in fastflow.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in fastflow.parameters())
    # print("Input : \n", np.array(input))
    # print("Kernel : \n", np.array(kernel))
    # input = torch.randn((batch_size, in_channels, image_size[1], image_size[2]))
    # input = torch.tensor(input, dtype=torch.float).cuda()
    # train_loader, val_loader, test_loader = load_data_cifar(data_aug=True, batch_size=batch_size)
    # for x, label in test_loader:
    #     input = x.cuda()
    #     break

    with torch.no_grad():
        input = torch.rand((1, *image_size)).to('cuda')
    print("Input", batch_size, input.shape[1:])
    B, C, H, W = input.shape
    # output = torch.zeros((B, C, H, W), dtype=torch.float).cuda()
    total_time_forward = 0
    total_time_reverse = 0
    n_loops = 10
    conv_output, _ = fastflow(input)
    torch.cuda.empty_cache()
    with torch.no_grad():
        for i in range(n_loops + 1):    
            # t = time.process_time()
            # conv_output, _ = fastflow(input)
            # forward_time = time.process_time() - t

            t = time.process_time()
            reverse_input, _ = fastflow.sample(n_samples=batch_size)
            reverse_time = time.process_time() - t
            if i != 0:
                # total_time_forward += forward_time
                total_time_reverse += reverse_time
            
    # t = time.process_time()
    # reverse_input_level2 = fastflow.reverse_level2(conv_output)
    # reverse_time_level2 = time.process_time() - t
    
    # error_level1 = torch.mean((input - reverse_input) ** 2)
    # error_level2 = torch.mean((input - reverse_input_level2) ** 2)

    # print(f"MSE Level 1 : {error_level1}")
    # print(f"MSE Level 2 : {error_level2}")
    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    # for param in fastflow.state_dict():
    #     print(param, torch.numel(fastflow.state_dict()[param]))
    # for p in fastflow.parameters():
    #     print(p.shape)
    # print(f"Average Forward Time : {total_time_forward / n_loops}s")
    print(f"Average Level 1 Reverse Time : {total_time_reverse / n_loops}")
    # print(f"Level 2 Reverse Time : {reverse_time_level2}s")

    return total_time_reverse / n_loops

def test_inverse_Emerging_MNIST(n_blocks=2, block_size=16, batch_size=100, image_size=(1, 28, 28)):

    glow = create_model_emerging(num_blocks=n_blocks,
                        block_size=block_size,
                        actnorm=True,
                        split_prior=True,
                        image_size=image_size).to('cuda')
    print("-----------------------------------------------------")
    in_channels = image_size[0]
    out_channels = in_channels

    total_params_learnable = sum(p.numel() for p in glow.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in glow.parameters())

    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    # for param in glow.state_dict(): 
    #     print(param, torch.numel(glow.state_dict()[param]))
    # for p in glow.parameters():
    #     print(p.shape)
    # print("Input : \n", np.array(input))
    # print("Kernel : \n", np.array(kernel))
    # input = torch.randn((batch_size, in_channels, image_size[1], image_size[2]))
    # input = torch.tensor(input, dtype=torch.float).cuda()
    train_loader, val_loader, test_loader = load_data(data_aug=False, batch_size=batch_size)
    for x, label in test_loader:
        input = x.cuda()
        break
    B, C, H, W = input.shape
    # output = torch.zeros((B, C, H, W), dtype=torch.float).cuda()
    total_time_forward = 0
    total_time_reverse = 0
    n_loops = 10
    with torch.no_grad():
        for i in range(n_loops + 1):    
            t = time.process_time()
            conv_output, _ = glow(input)
            forward_time = time.process_time() - t

            t = time.process_time()
            reverse_input, _ = glow.sample(n_samples=batch_size)
            reverse_time = time.process_time() - t
            if i != 0:
                total_time_forward += forward_time
                total_time_reverse += reverse_time

    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    
    print(f"Average Forward Time : {total_time_forward / n_loops}s")
    print(f"Average Level 1 Reverse Time : {total_time_reverse / n_loops}s")

def test_inverse_Emerging_CIFAR(n_blocks=3, block_size=32, batch_size=100, image_size=(3, 32, 32)):

    glow = create_model_emerging(num_blocks=n_blocks,
                        block_size=block_size,
                        actnorm=True,
                        split_prior=True,
                        image_size=image_size).to('cuda')
    print("-----------------------------------------------------")
    in_channels = image_size[0]
    out_channels = in_channels

    total_params_learnable = sum(p.numel() for p in glow.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in glow.parameters())

    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    # for param in glow.state_dict(): 
    #     print(param, torch.numel(glow.state_dict()[param]))
    # for p in glow.parameters():
    #     print(p.shape)
    # print("Input : \n", np.array(input))
    # print("Kernel : \n", np.array(kernel))
    # input = torch.randn((batch_size, in_channels, image_size[1], image_size[2]))
    # input = torch.tensor(input, dtype=torch.float).cuda()
    train_loader, val_loader, test_loader = load_data_cifar(data_aug=True, batch_size=batch_size)
    for x, label in test_loader:
        input = x.cuda()
        break
    B, C, H, W = input.shape
    # output = torch.zeros((B, C, H, W), dtype=torch.float).cuda()
    total_time_forward = 0
    total_time_reverse = 0
    n_loops = 10
    with torch.no_grad():
        for i in range(n_loops + 1):    
            t = time.process_time()
            conv_output, _ = glow(input)
            forward_time = time.process_time() - t

            t = time.process_time()
            reverse_input, _ = glow.sample(n_samples=batch_size)
            reverse_time = time.process_time() - t
            if i != 0:
                total_time_forward += forward_time
                total_time_reverse += reverse_time

    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    
    print(f"Average Forward Time : {total_time_forward / n_loops}s")
    print(f"Average Level 1 Reverse Time : {total_time_reverse / n_loops}s")

def test_inverse_Emerging_Imagenet32(n_blocks=3, block_size=48, batch_size=100, image_size=(3, 32, 32)):

    glow = create_model_emerging(num_blocks=n_blocks,
                        block_size=block_size,
                        actnorm=True,
                        split_prior=True,
                        image_size=image_size).to('cuda')
    print("-----------------------------------------------------")
    in_channels = image_size[0]
    out_channels = in_channels

    total_params_learnable = sum(p.numel() for p in glow.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in glow.parameters())

    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    # for param in glow.state_dict(): 
    #     print(param, torch.numel(glow.state_dict()[param]))
    # for p in glow.parameters():
    #     print(p.shape)
    # print("Input : \n", np.array(input))
    # print("Kernel : \n", np.array(kernel))
    # input = torch.randn((batch_size, in_channels, image_size[1], image_size[2]))
    # input = torch.tensor(input, dtype=torch.float).cuda()
    train_loader, val_loader, test_loader = load_data_imagenet(data_aug=True, 
                                                      batch_size=batch_size,
                                                      resolution=32,
                                                      data_dir=imagenet64_data_dir)
    for x, label in test_loader:
        input = x.cuda()
        break
    B, C, H, W = input.shape
    # output = torch.zeros((B, C, H, W), dtype=torch.float).cuda()
    total_time_forward = 0
    total_time_reverse = 0
    n_loops = 10
    with torch.no_grad():
        for i in range(n_loops + 1):    
            t = time.process_time()
            conv_output, _ = glow(input)
            forward_time = time.process_time() - t

            t = time.process_time()
            reverse_input, _ = glow.sample(n_samples=batch_size)
            reverse_time = time.process_time() - t
            if i != 0:
                total_time_forward += forward_time
                total_time_reverse += reverse_time

    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    
    print(f"Average Forward Time : {total_time_forward / n_loops}s")
    print(f"Average Level 1 Reverse Time : {total_time_reverse / n_loops}s")


def test_inverse_Emerging_Imagenet64(n_blocks=4, block_size=48, batch_size=100, image_size=(3, 64, 64)):

    glow = create_model_emerging(num_blocks=n_blocks,
                        block_size=block_size,
                        actnorm=True,
                        split_prior=True,
                        image_size=image_size).to('cuda')
    print("-----------------------------------------------------")
    in_channels = image_size[0]
    out_channels = in_channels

    total_params_learnable = sum(p.numel() for p in glow.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in glow.parameters())

    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    # for param in glow.state_dict(): 
    #     print(param, torch.numel(glow.state_dict()[param]))
    # for p in glow.parameters():
    #     print(p.shape)
    # print("Input : \n", np.array(input))
    # print("Kernel : \n", np.array(kernel))
    # input = torch.randn((batch_size, in_channels, image_size[1], image_size[2]))
    # input = torch.tensor(input, dtype=torch.float).cuda()
    train_loader, val_loader, test_loader = load_data_imagenet(data_aug=True, 
                                                      batch_size=batch_size,
                                                      resolution=64,
                                                      data_dir=imagenet64_data_dir)
    for x, label in test_loader:
        input = x.cuda()
        break
    B, C, H, W = input.shape
    # output = torch.zeros((B, C, H, W), dtype=torch.float).cuda()
    total_time_forward = 0
    total_time_reverse = 0
    n_loops = 10
    with torch.no_grad():
        for i in range(n_loops + 1):    
            t = time.process_time()
            conv_output, _ = glow(input)
            forward_time = time.process_time() - t

            t = time.process_time()
            reverse_input, _ = glow.sample(n_samples=batch_size)
            reverse_time = time.process_time() - t
            if i != 0:
                total_time_forward += forward_time
                total_time_reverse += reverse_time

    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    
    print(f"Average Forward Time : {total_time_forward / n_loops}s")
    print(f"Average Level 1 Reverse Time : {total_time_reverse / n_loops}s")

def test_timings_Emerging(n_blocks=3, block_size=32, batch_size=100, image_size=(3, 32, 32)):

    glow = create_model_emerging(num_blocks=n_blocks,
                        block_size=block_size,
                        actnorm=True,
                        split_prior=True,
                        image_size=image_size).to('cuda')
    glow.eval()
    print("-----------------------------------------------------")
    in_channels = image_size[0]
    out_channels = in_channels

    total_params_learnable = sum(p.numel() for p in glow.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in glow.parameters())

    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    # for param in glow.state_dict(): 
    #     print(param, torch.numel(glow.state_dict()[param]))
    # for p in glow.parameters():
    #     print(p.shape)
    # print("Input : \n", np.array(input))
    # print("Kernel : \n", np.array(kernel))
    # input = torch.randn((batch_size, in_channels, image_size[1], image_size[2]))
    # input = torch.tensor(input, dtype=torch.float).cuda()
    # train_loader, val_loader, test_loader = load_data_imagenet(data_aug=True, 
    #                                                   batch_size=batch_size,
    #                                                   resolution=64,
    #                                                   data_dir=imagenet64_data_dir)
    # for x, label in test_loader:
        # input = x.cuda()
        # break

    input = torch.rand((1, *image_size)).to('cuda')
    print("Input", input.shape)
    B, C, H, W = input.shape
    # output = torch.zeros((B, C, H, W), dtype=torch.float).cuda()
    total_time_forward = 0
    total_time_reverse = 0
    conv_output, _ = glow(input)
    n_loops = 10
    with torch.no_grad():
        for i in range(n_loops + 1):    
            # t = time.process_time()
            # conv_output, _ = glow(input)
            # forward_time = time.process_time() - t

            t = time.process_time()
            reverse_input, _ = glow.sample(n_samples=batch_size)
            reverse_time = time.process_time() - t
            if i != 0:
                # total_time_forward += forward_time
                total_time_reverse += reverse_time

    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    
    # print(f"Average Forward Time : {total_time_forward / n_loops}s")
    print(f"Average Level 1 Reverse Time : {total_time_reverse / n_loops}")
    return total_time_reverse / n_loops

def test_inverse_CINCFlow_CIFAR(n_blocks=3, block_size=32, batch_size=100, image_size=(3, 32, 32)):

    fastflow = create_model_cinc(num_blocks=n_blocks,
                        block_size=block_size,
                        actnorm=True,
                        split_prior=True,
                        image_size=image_size).to('cuda')
    fastflow.eval()
    print("-----------------------------------------------------")
    in_channels = image_size[0]
    out_channels = in_channels

    total_params_learnable = sum(p.numel() for p in fastflow.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in fastflow.parameters())
    # print("Input : \n", np.array(input))
    # print("Kernel : \n", np.array(kernel))
    # input = torch.randn((batch_size, in_channels, image_size[1], image_size[2]))
    # input = torch.tensor(input, dtype=torch.float).cuda()
    train_loader, val_loader, test_loader = load_data_cifar(data_aug=True, batch_size=batch_size)
    for x, label in test_loader:
        input = x.cuda()
        break
    print("Input", x.shape)
    B, C, H, W = input.shape
    # output = torch.zeros((B, C, H, W), dtype=torch.float).cuda()
    total_time_forward = 0
    total_time_reverse = 0
    n_loops = 10
    with torch.no_grad():
        for i in range(n_loops + 1):    
            t = time.process_time()
            conv_output, _ = fastflow(input)
            forward_time = time.process_time() - t

            t = time.process_time()
            reverse_input, _ = fastflow.sample(n_samples=batch_size)
            reverse_time = time.process_time() - t
            if i != 0:
                total_time_forward += forward_time
                total_time_reverse += reverse_time

    # t = time.process_time()
    # reverse_input_level2 = fastflow.reverse_level2(conv_output)
    # reverse_time_level2 = time.process_time() - t
    
    # error_level1 = torch.mean((input - reverse_input) ** 2)
    # error_level2 = torch.mean((input - reverse_input_level2) ** 2)

    # print(f"MSE Level 1 : {error_level1}")
    # print(f"MSE Level 2 : {error_level2}")
    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    # for param in fastflow.state_dict():
    #     print(param, torch.numel(fastflow.state_dict()[param]))
    # for p in fastflow.parameters():
    #     print(p.shape)
    print(f"Average Forward Time : {total_time_forward / n_loops}s")
    print(f"Average Level 1 Reverse Time : {total_time_reverse / n_loops}s")
    # print(f"Level 2 Reverse Time : {reverse_time_level2}s")

def test_inverse_CINCFlow_Imagenet32(n_blocks=3, block_size=48, batch_size=100, image_size=(3, 32, 32)):

    fastflow = create_model_cinc(num_blocks=n_blocks,
                        block_size=block_size,
                        actnorm=True,
                        split_prior=True,
                        image_size=image_size).to('cuda')
    fastflow.eval()
    print("-----------------------------------------------------")
    in_channels = image_size[0]
    out_channels = in_channels

    total_params_learnable = sum(p.numel() for p in fastflow.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in fastflow.parameters())
    # print("Input : \n", np.array(input))
    # print("Kernel : \n", np.array(kernel))
    # input = torch.randn((batch_size, in_channels, image_size[1], image_size[2]))
    # input = torch.tensor(input, dtype=torch.float).cuda()
    train_loader, val_loader, test_loader = load_data_imagenet(data_aug=True, 
                                                      batch_size=batch_size,
                                                      resolution=32,
                                                      data_dir=imagenet64_data_dir)
    for x, label in test_loader:
        input = x.cuda()
        break
    print("Input", x.shape)
    B, C, H, W = input.shape
    # output = torch.zeros((B, C, H, W), dtype=torch.float).cuda()
    total_time_forward = 0
    total_time_reverse = 0
    n_loops = 10
    with torch.no_grad():
        for i in range(n_loops + 1):    
            t = time.process_time()
            conv_output, _ = fastflow(input)
            forward_time = time.process_time() - t

            t = time.process_time()
            reverse_input, _ = fastflow.sample(n_samples=batch_size)
            reverse_time = time.process_time() - t
            if i != 0:
                total_time_forward += forward_time
                total_time_reverse += reverse_time

    # t = time.process_time()
    # reverse_input_level2 = fastflow.reverse_level2(conv_output)
    # reverse_time_level2 = time.process_time() - t
    
    # error_level1 = torch.mean((input - reverse_input) ** 2)
    # error_level2 = torch.mean((input - reverse_input_level2) ** 2)

    # print(f"MSE Level 1 : {error_level1}")
    # print(f"MSE Level 2 : {error_level2}")
    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    # for param in fastflow.state_dict():
    #     print(param, torch.numel(fastflow.state_dict()[param]))
    # for p in fastflow.parameters():
    #     print(p.shape)
    print(f"Average Forward Time : {total_time_forward / n_loops}")
    print(f"Average Level 1 Reverse Time : {total_time_reverse / n_loops}")


def test_inverse_CINCFlow_Imagenet64(n_blocks=4, block_size=48, batch_size=100, image_size=(3, 64, 64)):

    fastflow = create_model_cinc(num_blocks=n_blocks,
                        block_size=block_size,
                        actnorm=True,
                        split_prior=True,
                        image_size=image_size).to('cuda')
    fastflow.eval()
    print("-----------------------------------------------------")
    in_channels = image_size[0]
    out_channels = in_channels

    total_params_learnable = sum(p.numel() for p in fastflow.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in fastflow.parameters())
    # print("Input : \n", np.array(input))
    # print("Kernel : \n", np.array(kernel))
    # input = torch.randn((batch_size, in_channels, image_size[1], image_size[2]))
    # input = torch.tensor(input, dtype=torch.float).cuda()
    train_loader, val_loader, test_loader = load_data_imagenet(data_aug=True, 
                                                      batch_size=batch_size,
                                                      resolution=64,
                                                      data_dir=imagenet64_data_dir)
    for x, label in test_loader:
        input = x.cuda()
        break
    print("Input", x.shape)
    B, C, H, W = input.shape
    # output = torch.zeros((B, C, H, W), dtype=torch.float).cuda()
    total_time_forward = 0
    total_time_reverse = 0
    n_loops = 10
    with torch.no_grad():
        for i in range(n_loops + 1):    
            t = time.process_time()
            conv_output, _ = fastflow(input)
            forward_time = time.process_time() - t

            t = time.process_time()
            reverse_input, _ = fastflow.sample(n_samples=batch_size)
            reverse_time = time.process_time() - t
            if i != 0:
                total_time_forward += forward_time
                total_time_reverse += reverse_time

    # t = time.process_time()
    # reverse_input_level2 = fastflow.reverse_level2(conv_output)
    # reverse_time_level2 = time.process_time() - t
    
    # error_level1 = torch.mean((input - reverse_input) ** 2)
    # error_level2 = torch.mean((input - reverse_input_level2) ** 2)

    # print(f"MSE Level 1 : {error_level1}")
    # print(f"MSE Level 2 : {error_level2}")
    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    # for param in fastflow.state_dict():
    #     print(param, torch.numel(fastflow.state_dict()[param]))
    # for p in fastflow.parameters():
    #     print(p.shape)
    print(f"Average Forward Time : {total_time_forward / n_loops}")
    print(f"Average Level 1 Reverse Time : {total_time_reverse / n_loops}")

def test_timings_CINC(n_blocks=3, block_size=32, batch_size=100, image_size=(3, 32, 32)):

    # fastflow = FastFlow_CIFAR(n_blocks=n_blocks,
    #                     block_size=block_size,
    #                     actnorm=True,
    #                     image_size=image_size).to('cuda')
    fastflow = create_model_cinc(num_blocks=n_blocks,
                        block_size=block_size,
                        actnorm=True,
                        split_prior=True,
                        image_size=image_size).to('cuda')
    fastflow.eval()
    print("-----------------------------------------------------")
    in_channels = image_size[0]
    out_channels = in_channels

    total_params_learnable = sum(p.numel() for p in fastflow.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in fastflow.parameters())
    # print("Input : \n", np.array(input))
    # print("Kernel : \n", np.array(kernel))
    # input = torch.randn((batch_size, in_channels, image_size[1], image_size[2]))
    # input = torch.tensor(input, dtype=torch.float).cuda()
    # train_loader, val_loader, test_loader = load_data_cifar(data_aug=True, batch_size=batch_size)
    # for x, label in test_loader:
    #     input = x.cuda()
    #     break

    input = torch.rand((1, *image_size)).to('cuda')
    print("Input", input.shape)
    B, C, H, W = input.shape
    # output = torch.zeros((B, C, H, W), dtype=torch.float).cuda()
    total_time_forward = 0
    total_time_reverse = 0
    n_loops = 10
    conv_output, _ = fastflow(input)
    torch.cuda.empty_cache()
    with torch.no_grad():
        for i in range(n_loops + 1):    
            # t = time.process_time()
            # conv_output, _ = fastflow(input)
            # forward_time = time.process_time() - t

            t = time.process_time()
            reverse_input, _ = fastflow.sample(n_samples=batch_size)
            reverse_time = time.process_time() - t
            if i != 0:
                # total_time_forward += forward_time
                total_time_reverse += reverse_time
            
    # t = time.process_time()
    # reverse_input_level2 = fastflow.reverse_level2(conv_output)
    # reverse_time_level2 = time.process_time() - t
    
    # error_level1 = torch.mean((input - reverse_input) ** 2)
    # error_level2 = torch.mean((input - reverse_input_level2) ** 2)

    # print(f"MSE Level 1 : {error_level1}")
    # print(f"MSE Level 2 : {error_level2}")
    print("Total Params: ", total_params)
    print("Total Params(Learnable): ", total_params_learnable)
    # for param in fastflow.state_dict():
    #     print(param, torch.numel(fastflow.state_dict()[param]))
    # for p in fastflow.parameters():
    #     print(p.shape)
    # print(f"Average Forward Time : {total_time_forward / n_loops}s")
    print(f"Average Level 1 Reverse Time : {total_time_reverse / n_loops}")
    # print(f"Level 2 Reverse Time : {reverse_time_level2}s")
    return total_time_reverse / n_loops
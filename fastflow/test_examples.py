import torch
import numpy as np
from test_layers import *


# test_PaddedConv2d(
#     x = torch.tensor(
#     [
#         [1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]
#     ], dtype=torch.float32
#     ).reshape(1, 1, 3, 3), 

#     w = torch.tensor(                             #Kernel has to be properly set
#         [
#             [0, 1],
#             [1, 0]
#         ], dtype=torch.float32
#     ).reshape(1, 1, 2, 2),

#     order='BL',

#     print_answer=False
# )

# test_clear_grad(
#     n_channels=3,
#     kernel_size=(2, 2)
# )

# test_PaddedConv2d(
#     x = torch.tensor(
#     [
#         [1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]
#     ], dtype=torch.float32
#     ).reshape(1, 1, 3, 3), 

#     # w = torch.tensor(                           #Kernel has to be properly set
#     #     [
#     #         [1, 0],
#     #         [0, 2]
#     #     ], dtype=torch.float32
#     # ).reshape(1, 1, 2, 2),
#     w=None,
#     order='BR',

#     print_answer=False
# )


# test_PaddedConv2d(
#     x = torch.tensor(
#     [
#         [ 1.,  2.,  3.],
#         [ 8., 11.,  6.],
#         [17., 20.,  9.]
#     ], dtype=torch.float32
#     ).reshape(1, 1, 3, 3), 

#     w = torch.tensor(                                #Kernel has to be properly set
#         [
#             [0, 2],
#             [1, 0]
#         ], dtype=torch.float32
#     ).reshape(1, 1, 2, 2),

#     order='TR',
#     is_input=False,
#     print_answer=False
# )

# test_FastFlowUnit(
#     x = torch.rand((1, 16, 7, 7)),
#     kernel_size=(2, 2),
#     block_size=20,
#     is_input=True,
#     print_answer=False
# )

# test_FastFlowUnit(
#     x = torch.rand((10, 48, 100, 100)),
#     kernel_size=(3, 3),
#     block_size=20,
#     is_input=True,
#     print_answer=False
# )

# x = torch.randn(28 * 28).reshape(-1, 28, 28)
# test_Dequantization(
#     x=x,
#     size=x.shape,
#     is_input=False
# )

# test_Normalization(
#     x = torch.rand((10, 48, 10, 10)),
#     translation=10,
#     scale=0.1,
#     is_input=False
# )


# alpha = 1e-6
# test_Normalization(
#     x = torch.rand((10, 48, 10, 10)),
#     translation=-alpha,
#     scale=1 / (1 - 2 * alpha),
#     is_input=False
# )

# test_LogitTransform(
#     x = torch.rand((10, 48, 100, 100))
# )

# test_LogitTransform(
#     x = torch.rand((10, 16, 7, 7)),
#     is_input=False
# )

# test_Squeeze(
#     x = torch.rand((100, 16, 20, 10)),       #H, W have to be even
#     is_input=False
# )

# test_ActNorm(
#     x=torch.rand((100, 1, 13, 18))
# )

# test_ActNorm(
#     x=torch.rand((30, 25, 100, 100)),  #Cannot do this because the layer has to be initialized

#     is_input=False
# )


# test_Conv1x1(
#     x=torch.rand((100, 10, 13, 180))
# )

# test_Conv1x1(
#     x=torch.rand((100, 10, 13, 180)),
#     is_input=False
# )

# test_Coupling(
#     x=torch.rand((10, 8, 100, 100)),
#     is_input=False
# )


# test_SplitPrior(
#     x=torch.rand((1, 2, 3, 3)),             #Cannot be tested
#     print_answer=True
# )

# test_Split(
#     x=torch.rand((1, 2, 3, 3)),
#     is_input=False,                          #Cannot be tested
#     print_answer=True
# )

# test_FastFlowMNIST(
#     # /home/aditya.kallappa/Research/NormalizingFlows/FastFlow/fastflow
#     # checkpoint_path="./wandb/run-20220329_113439-39od8z5l/files/checkpoint.tar",
#     true_inverse=False,
#     batch_size=100,
#     plot=True
# )

# test_GlowMNIST(
#     checkpoint_path='./wandb/checkpoint.tar'
# )

# test_FastFlow(
#     # /home/aditya.kallappa/Research/NormalizingFlows/FastFlow/fastflow
#     checkpoint_path="./wandb/3L-32K_FastFlow_CIFAR_Epoch_14.tar",
#     true_inverse=True,
#     batch_size=100,
#     plot=True
# )

# test_Preprocess(
#     x=torch.rand((10, 20, 50, 100)),         #cannot be tested with is_input=False
#     is_input=False
# )

# test_GlowStep(
#     x=torch.rand((10, 20, 50, 100)),         #cannot be tested with is_input=False
#     is_input=True
# )

# test_FastFlowStep(
#     x=torch.rand((10, 20, 50, 100)),         #cannot be tested with is_input=False
#     is_input=True
# )


# test_Split(
#     x=torch.rand((10, 20, 25, 25)),         #cannot be tested with is_input=False
#     is_input=True
# )

# test_Gaussianize(
#     x=torch.rand((10, 20, 25, 25)),         #cannot be tested with is_input=False
#     is_input=True,
#     with_zeros=True
# )

# test_FastFlowLevel(
#     x=torch.rand((10, 8, 32, 32)),         #cannot be tested with is_input=False
#     is_input=True
# )


# test_inverse_PaddedConv2d(in_channels=1, out_channels=None, kernel_size=(3, 3), bias=True, order='TL', batch_size=1, image_size=(28, 28))
# test_inverse_PaddedConv2d(in_channels=3, out_channels=None, kernel_size=(3, 3), bias=True, order='TR', batch_size=1, image_size=(32, 32))
# test_inverse_PaddedConv2d(in_channels=10, out_channels=None, kernel_size=(3, 3), bias=True, order='BL', batch_size=10, image_size=(50, 50))
# test_inverse_PaddedConv2d(in_channels=20, out_channels=None, kernel_size=(5, 5), bias=True, order='BR', batch_size=100, image_size=(128, 128))
# test_inverse_PaddedConv2d(in_channels=50, out_channels=None, kernel_size=(3, 3), bias=True, order='TL', batch_size=10, image_size=(256, 256))

# test_SigmoidFlow(
#     x = torch.randn(1, 12, 32, 32)
# )


# test_FastFlowSigmoid(
#     n_blocks=3,
#     block_size=32,
#     batch_size=64,
#     image_size=(3, 32, 32)
# )



# test_inverse_Glow_MNIST(n_blocks=2, block_size=16, batch_size=100, image_size=(1, 28, 28))
# test_inverse_Glow_MNIST(n_blocks=2, block_size=16, batch_size=1, image_size=(1, 28, 28))
# test_inverse_Glow_CIFAR(n_blocks=3, block_size=32, batch_size=100, image_size=(3, 32, 32))
# test_inverse_Glow_Imagenet32(n_blocks=3, block_size=48, batch_size=100, image_size=(3, 32, 32))
# test_inverse_Glow_Imagenet64(n_blocks=4, block_size=48, batch_size=100, image_size=(3, 64, 64))
# test_inverse_FastFlow_MNIST(n_blocks=2, block_size=16, batch_size=1, image_size=(1, 28, 28))
# test_inverse_FastFlow_MNIST(n_blocks=2, block_size=16, batch_size=50, image_size=(1, 28, 28))
# test_inverse_FastFlow_CIFAR2(n_blocks=3, block_size=32, batch_size=100, image_size=(3, 32, 32))
# test_inverse_FastFlow_CIFAR(n_blocks=3, block_size=32, batch_size=100, image_size=(3, 32, 32))
# test_inverse_FastFlow_CIFAR(n_blocks=3, block_size=48, batch_size=1, image_size=(3, 32, 32))
# test_inverse_FastFlow_CIFAR(n_blocks=3, block_size=48, batch_size=100, image_size=(3, 32, 32))


# test_inverse_FastFlow_Imagenet64(n_blocks=4, block_size=48, batch_size=100, image_size=(3, 64, 64))
# test_inverse_FastFlow_Imagenet32(n_blocks=3, block_size=48, batch_size=100, image_size=(3, 32, 32))
# test_inverse_FastFlow_Imagenet64(n_blocks=4, block_size=50, batch_size=100, image_size=(3, 64, 64))

# test_inverse_FastFlow_CIFAR(n_blocks=3, block_size=32, batch_size=1, image_size=(3, 32, 32))
# test_inverse_FastFlow_CIFAR_modified(n_blocks=4, block_size=6, T=8, batch_size=1, image_size=(3, 32, 32))
# test_inverse_Glow_MNIST(n_blocks=3, block_size=32, batch_size=1, image_size=(3, 32, 32))
# test_inverse_SNF_MNIST()
# test_inverse_SNF_CIFAR()
# test_inverse_SNF_Imagenet32()
# test_inverse_SNF_Imagenet64()

batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

# for b in batch_sizes:
#     answer = []
#     answer.append(test_timings(n_blocks=2, block_size=10, batch_size=b, image_size=(2, 16, 16)))
#     answer.append(test_timings(n_blocks=2, block_size=10, batch_size=b, image_size=(2, 16, 32)))
#     answer.append(test_timings(n_blocks=2, block_size=10, batch_size=b, image_size=(2, 32, 32)))
#     answer.append(test_timings(n_blocks=2, block_size=10, batch_size=b, image_size=(2, 32, 64)))
#     answer.append(test_timings(n_blocks=2, block_size=10, batch_size=b, image_size=(2, 64, 64)))
#     answer.append(test_timings(n_blocks=2, block_size=10, batch_size=b, image_size=(2, 64, 128)))
#     answer.append(test_timings(n_blocks=2, block_size=10, batch_size=b, image_size=(2, 128, 128)))
#     # answer.append(test_timings(n_blocks=2, block_size=16, batch_size=b, image_size=(1, 256, 256)))
#     # answer.append(test_timings(n_blocks=2, block_size=16, batch_size=b, image_size=(3, 32, 64))
#     # answer.append(test_timings(n_blocks=2, block_size=16, batch_size=b, image_size=(3, 64, 64))
#     print("*"*100)
#     # answer.append(test_timings(n_blocks=2, block_size=50, batch_size=b, image_size=(3, 8, 8))
#     # answer.append(test_timings(n_blocks=2, block_size=50, batch_size=b, image_size=(3, 8, 16))
#     # answer.append(test_timings(n_blocks=2, block_size=50, batch_size=b, image_size=(3, 16, 16))
#     # answer.append(test_timings(n_blocks=2, block_size=50, batch_size=b, image_size=(3, 16, 32))
#     # answer.append(test_timings(n_blocks=2, block_size=50, batch_size=b, image_size=(3, 32, 32))
#     # answer.append(test_timings(n_blocks=2, block_size=50, batch_size=b, image_size=(3, 32, 64))
#     # answer.append(test_timings(n_blocks=2, block_size=50, batch_size=b, image_size=(3, 64, 64))
#     np_answer = np.array(answer)
#     np.save(f'numpy_files/fastflow_{b}.npy', np_answer)

for b in batch_sizes:
    answer = []
    answer.append(test_timings_CINC(n_blocks=2, block_size=10, batch_size=b, image_size=(2, 16, 16)))
    answer.append(test_timings_CINC(n_blocks=2, block_size=10, batch_size=b, image_size=(2, 16, 32)))
    answer.append(test_timings_CINC(n_blocks=2, block_size=10, batch_size=b, image_size=(2, 32, 32)))
    answer.append(test_timings_CINC(n_blocks=2, block_size=10, batch_size=b, image_size=(2, 32, 64)))
    answer.append(test_timings_CINC(n_blocks=2, block_size=10, batch_size=b, image_size=(2, 64, 64)))
    answer.append(test_timings_CINC(n_blocks=2, block_size=10, batch_size=b, image_size=(2, 64, 128)))
    answer.append(test_timings_CINC(n_blocks=2, block_size=10, batch_size=b, image_size=(2, 128, 128)))
    # answer.append(test_timings(n_blocks=2, block_size=16, batch_size=b, image_size=(1, 256, 256)))
    # answer.append(test_timings(n_blocks=2, block_size=16, batch_size=b, image_size=(3, 32, 64))
    # answer.append(test_timings(n_blocks=2, block_size=16, batch_size=b, image_size=(3, 64, 64))
    print("*"*100)
    # answer.append(test_timings(n_blocks=2, block_size=50, batch_size=b, image_size=(3, 8, 8))
    # answer.append(test_timings(n_blocks=2, block_size=50, batch_size=b, image_size=(3, 8, 16))
    # answer.append(test_timings(n_blocks=2, block_size=50, batch_size=b, image_size=(3, 16, 16))
    # answer.append(test_timings(n_blocks=2, block_size=50, batch_size=b, image_size=(3, 16, 32))
    # answer.append(test_timings(n_blocks=2, block_size=50, batch_size=b, image_size=(3, 32, 32))
    # answer.append(test_timings(n_blocks=2, block_size=50, batch_size=b, image_size=(3, 32, 64))
    # answer.append(test_timings(n_blocks=2, block_size=50, batch_size=b, image_size=(3, 64, 64))
    np_answer = np.array(answer)
    np.save(f'numpy_files/cinc_{b}.npy', np_answer)

# for b in batch_sizes:
#     answer = []
#     answer.append(test_timings_Emerging(n_blocks=2, block_size=10, batch_size=b, image_size=(2, 16, 16)))
#     answer.append(test_timings_Emerging(n_blocks=2, block_size=10, batch_size=b, image_size=(2, 16, 32)))
#     answer.append(test_timings_Emerging(n_blocks=2, block_size=10, batch_size=b, image_size=(2, 32, 32)))
#     answer.append(test_timings_Emerging(n_blocks=2, block_size=10, batch_size=b, image_size=(2, 32, 64)))
#     answer.append(test_timings_Emerging(n_blocks=2, block_size=10, batch_size=b, image_size=(2, 64, 64)))
#     answer.append(test_timings_Emerging(n_blocks=2, block_size=10, batch_size=b, image_size=(2, 64, 128)))
#     answer.append(test_timings_Emerging(n_blocks=2, block_size=10, batch_size=b, image_size=(2, 128, 128)))
#     # answer.append(test_timings(n_blocks=2, block_size=16, batch_size=b, image_size=(1, 256, 256)))
#     # answer.append(test_timings(n_blocks=2, block_size=16, batch_size=b, image_size=(3, 32, 64))
#     # answer.append(test_timings(n_blocks=2, block_size=16, batch_size=b, image_size=(3, 64, 64))
#     print("*"*100)
# #     # answer.append(test_timings(n_blocks=2, block_size=50, batch_size=b, image_size=(3, 8, 8))
# #     # answer.append(test_timings(n_blocks=2, block_size=50, batch_size=b, image_size=(3, 8, 16))
# #     # answer.append(test_timings(n_blocks=2, block_size=50, batch_size=b, image_size=(3, 16, 16))
# #     # answer.append(test_timings(n_blocks=2, block_size=50, batch_size=b, image_size=(3, 16, 32))
# #     # answer.append(test_timings(n_blocks=2, block_size=50, batch_size=b, image_size=(3, 32, 32))
# #     # answer.append(test_timings(n_blocks=2, block_size=50, batch_size=b, image_size=(3, 32, 64))
# #     # answer.append(test_timings(n_blocks=2, block_size=50, batch_size=b, image_size=(3, 64, 64))
#     np_answer = np.array(answer)
#     np.save(f'numpy_files/emerging_{b}.npy', np_answer)

# torch.cuda.empty_cache()
# test_timings(n_blocks=2, block_size=16, batch_size=1, image_size=(1, 128, 128))
# torch.cuda.empty_cache()
# test_timings(n_blocks=2, block_size=16, batch_size=1, image_size=(1, 256, 256))

# test_inverse_Emerging_MNIST()
# test_inverse_Emerging_CIFAR()
# test_inverse_Emerging_Imagenet32()
# test_inverse_Emerging_Imagenet64()
# print("*"*100)
# test_timings_Emerging(n_blocks=2, block_size=16, batch_size=100, image_size=(3, 8, 8))
# test_timings_Emerging(n_blocks=2, block_size=16, batch_size=100, image_size=(3, 8, 16))
# test_timings_Emerging(n_blocks=2, block_size=16, batch_size=100, image_size=(3, 16, 16))
# test_timings_Emerging(n_blocks=2, block_size=16, batch_size=100, image_size=(3, 16, 32))
# test_timings_Emerging(n_blocks=2, block_size=16, batch_size=100, image_size=(3, 32, 32))
# test_timings_Emerging(n_blocks=2, block_size=16, batch_size=100, image_size=(3, 32, 64))
# test_timings_Emerging(n_blocks=2, block_size=16, batch_size=100, image_size=(3, 64, 64))

# print("*"*100)

# test_timings_Emerging(n_blocks=2, block_size=48, batch_size=100, image_size=(3, 8, 8))
# test_timings_Emerging(n_blocks=2, block_size=48, batch_size=100, image_size=(3, 8, 16))
# test_timings_Emerging(n_blocks=2, block_size=48, batch_size=100, image_size=(3, 16, 16))
# test_timings_Emerging(n_blocks=2, block_size=48, batch_size=100, image_size=(3, 16, 32))
# test_timings_Emerging(n_blocks=2, block_size=48, batch_size=100, image_size=(3, 32, 32))
# test_timings_Emerging(n_blocks=2, block_size=48, batch_size=100, image_size=(3, 32, 64))
# test_timings_Emerging(n_blocks=2, block_size=48, batch_size=100, image_size=(3, 64, 64))


# test_inverse_CINCFlow_CIFAR()
# test_inverse_CINCFlow_Imagenet32()
# test_inverse_CINCFlow_Imagenet64()

# test_timings_CINC(n_blocks=2, block_size=16, batch_size=100, image_size=(3, 8, 8))
# test_timings_CINC(n_blocks=2, block_size=16, batch_size=100, image_size=(3, 8, 16))
# test_timings_CINC(n_blocks=2, block_size=16, batch_size=100, image_size=(3, 16, 16))
# test_timings_CINC(n_blocks=2, block_size=16, batch_size=100, image_size=(3, 16, 32))
# test_timings_CINC(n_blocks=2, block_size=16, batch_size=100, image_size=(3, 32, 32))
# test_timings_CINC(n_blocks=2, block_size=16, batch_size=100, image_size=(3, 32, 64))
# test_timings_CINC(n_blocks=2, block_size=16, batch_size=100, image_size=(3, 64, 64))
# print("*"*100)
# test_timings_CINC(n_blocks=2, block_size=50, batch_size=100, image_size=(3, 8, 8))
# test_timings_CINC(n_blocks=2, block_size=50, batch_size=100, image_size=(3, 8, 16))
# test_timings_CINC(n_blocks=2, block_size=50, batch_size=100, image_size=(3, 16, 16))
# test_timings_CINC(n_blocks=2, block_size=50, batch_size=100, image_size=(3, 16, 32))
# test_timings_CINC(n_blocks=2, block_size=50, batch_size=100, image_size=(3, 32, 32))
# test_timings_CINC(n_blocks=2, block_size=50, batch_size=100, image_size=(3, 32, 64))
# test_timings_CINC(n_blocks=2, block_size=50, batch_size=100, image_size=(3, 64, 64))

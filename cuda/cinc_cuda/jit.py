import numpy as np

np.set_printoptions(precision=0,suppress=True, linewidth=120, )

from torch.utils.cpp_extension import load
import torch

cinc_cuda = load(
    'cinc_cuda', ['cinc_cuda.cpp', 'cinc_cuda_kernel.cu'], verbose=True)
# help(cinc_cuda)

def test_inverse(input, kernel):
    
    print("------------------------------------")
    
    print("Input : \n", np.array(input))
    print("Kernel : \n", np.array(kernel))

    input = torch.tensor(input, dtype=torch.float).cuda()
    kernel = torch.tensor(kernel, dtype=torch.float).cuda()
    
    n = input.size()[0]
    k = kernel.size()[0]
    
    output = torch.zeros((n,n), dtype=torch.float).cuda()
    
    cinc_cuda.inverse(input, kernel, output)
    
    print("Output : \n", output.cpu().numpy())

    # compute convolution of output with kernel and see if we get the input back
    error = (input - torch.nn.functional.conv2d(torch.nn.functional.pad(output.reshape(1,1,n,n), (k-1,0,k-1,0), "constant", 0), kernel.reshape(1,1,k,k))[0,0]).abs().sum().item()

    print("Error : ", error )

    

test_inverse(
    input=[
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,12],
            [13,14,15,16]
        ],
    kernel=[
            [0,0],
            [0,1]
    ])
    

test_inverse(
    input=[
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,12],
            [13,14,15,16]
        ],
    kernel=[
            [1,0],
            [0,1]
    ])


test_inverse(
    input=[
            [1,2,3],
            [5,6,7],
            [9,10,11]
        ],
    kernel=[
            [1,0],
            [0,1]
    ])


test_inverse(
    input=[
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,12],
            [13,14,15,16]
        ],
    kernel=[
            [1,0,0],
            [0,1,0],
            [0,0,1],
    ])

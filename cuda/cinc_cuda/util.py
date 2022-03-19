from torch.nn.functional import conv2d, pad
import numpy as np
np.set_printoptions(precision=10,suppress=True, linewidth=120, )
from torch.utils.cpp_extension import load
import torch

import time

cinc_cuda = load(
    'cinc_cuda', ['cinc_cuda_general.cpp', 'cinc_cuda_kernel_general.cu'], verbose=True)
# help(cinc_cuda)


def test_inverse(input, kernel):
    
    print("-----------------------------------------------------")
    
    print("Input : \n", np.array(input))
    print("Kernel : \n", np.array(kernel))

    input = torch.tensor(input, dtype=torch.float).cuda()
    kernel = torch.tensor(kernel, dtype=torch.float).cuda()
    
    m = input.size()[0]
    n = input.size()[1]
    k = kernel.size()[0]
    
    output = torch.zeros((m,n,n), dtype=torch.float).cuda()
    
    t = time.process_time()
    cinc_cuda.inverse(input, kernel, output)
    t = time.process_time() - t
    print("Output : \n", output.cpu().numpy())

    # compute convolution of output with kernel and see if we get the input back
    error = (input.reshape(m,1,n,n) - conv2d(pad(output.reshape(m,1,n,n), (k-1,0,k-1,0), "constant", 0), kernel.reshape(1,1,k,k))).abs().sum().item()

    print(f"Error : {error}")
    print(f"Time : {t}s")

    
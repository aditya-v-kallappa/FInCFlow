from numpy import dtype
from torch.utils.cpp_extension import load
import torch

cinc_cuda = load(
    'cinc_cuda', ['cinc_cuda.cpp', 'cinc_cuda_kernel.cu'], verbose=True)
# help(cinc_cuda)


input = torch.tensor([[1,2,3,4],
                      [5,6,7,8],
                      [9,10,11,12],
                      [13,14,15,16]
                      ], dtype=torch.float).cuda()


kernel = torch.tensor([[1,0],
                       [0,1]], dtype=torch.float).cuda()



output = torch.zeros((4,4), dtype=torch.float).cuda()

cinc_cuda.inverse(input, kernel, output)

print(output)
from torch.utils.cpp_extension import load
cinc_cuda = load(
    'cinc_cuda', ['cinc_cuda.cpp', 'cinc_cuda_kernel.cu'], verbose=True)
help(cinc_cuda)

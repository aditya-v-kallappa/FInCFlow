# Cuda implementation for inverse

For now only the boilerplate code is added. Inverse has to be implemented in the cinc_cuda_kernel.cu file (inside function cinc_cuda_inverse_kernel).


## Setup
1. download the ninja build binary (https://github.com/ninja-build/ninja/releases/download/v1.10.2/ninja-linux.zip) and put it under misc/bin (under home folder) 
2. install pytorch lts (1.8.2), with cuda 10.2
3. source the env.sh file
4. in the cinc_cuda directory, run python test_cuda_kernel.py
5. from cinc_cuda import inverse

## References
- https://pytorch.org/tutorials/advanced/cpp_extension.html
- https://developer.nvidia.com/blog/even-easier-introduction-cuda/
- https://github.com/pytorch/extension-cpp

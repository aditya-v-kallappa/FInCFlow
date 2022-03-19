#include <torch/extension.h>

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {


// Implementation for batch size = 1, channels = 1
template <typename scalar_t>
__global__ void cinc_cuda_inverse_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> kernel,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> output, int k, int d) {
  
    int tid = blockIdx.x * blockDim.x + threadIdx.x;   // thread id
    const auto m  = output.size(0); // assuming height and width to be the same
    const auto n  = output.size(1);
    // NOTE: For some reason kernel.size(0) doesnot work correctly here

    // Compute the index from the dth diagonal assigned to the thread
    int h, i, j;
    h = tid % m; // batch index encoded in lower indices modulo batchsize
    tid = (tid - h) / m; // remaining part of tid has the index along the diagonal
    if (d <= n) {
        i = d - 1 - tid;
        j = tid;
    } 
    else {
        j = (d - n)  + tid;
        i = n - 1 - tid;
    }

    // compute entry of the output in the diagonal d assigned to this thread
    output[h][i][j] = input[h][i][j];
    for (int a = 0; a < k; a++) {
        for (int b = 0; b < k; b++) {
            if ( i-(k-1)+a >=0 && j-(k-1)+b >=0 && !((a==k-1) && (b==k-1))) {
            // # if __CUDA_ARCH__ >= 200
            //   printf("%d %d %d %d %d %d \n",i,j,a,b,n,k);
            // #endif
                output[h][i][j] -= kernel[a][b]*output[h][i-(k-1)+a][j-(k-1)+b];
            }
        }
    }
}

} // namespace

std::vector<torch::Tensor> cinc_cuda_inverse(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor output) {

  // assuming batch size 1 for initial implementation
  // all the tensors above are 2D only

  const auto m = output.size(0);
  const auto n = output.size(1); // assuming height and width to be the same
  const auto k = kernel.size(0); // assuming square kernel

  // printf("%d %d", n,k);

  for (int d = 1; d <= 2*n-1; d++) { // Iterating over diagonal index

    int max_threads = 1024;

    // all elements of the dth diagonal computed in parallel
    int threads = d;
    if (d > n) {
      threads = 2*n-d;
    }
    threads = threads*m;
    
    const int blocks = (max_threads +threads)/max_threads; // use multiple blocks if 2d-1 > threads

    AT_DISPATCH_FLOATING_TYPES(input.type(), "cinc_inverse_cuda", ([&] {
      cinc_cuda_inverse_kernel<scalar_t><<<blocks, threads>>>(
          input.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
          kernel.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          output.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(), k, d);
    }));

    // synchronize all threads
    cudaDeviceSynchronize();

  }

  return {output};
}

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

template <typename scalar_t>
__global__ void cinc_cuda_inverse_kernel(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> kernel,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output, int d) {
  
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;   // thread id
  const auto n  = output.size(0); // assuming height and width to be the same
  const auto k  = kernel.size(0); // assuming square kernel

  if (tid < n) { // if we allocated more threads, exess threads dont do anything

    // Compute the index from the dth diagonal assigned to the thread
    int i, j;
    if (d <= n) {
      i = d - 1 - tid;
      j = tid;
    } 
    else {
      j = (d - n)  + tid;
      i = n - 1 - tid;
    }

    // compute entry of the output in the diagonal d assigned to this thread
    output[i][j] = input[i][j];
    for (int a = 0; a < k; a++) {
       for (int b = 0; b < k; b++) {
          if ( i-(k-1)+a >=0 && j-(k-1)+b >=0 && !((a==k-1) && (b==k-1))) {
            output[i][j] -= kernel[a][b]*output[i-(k-1)+a][j-(k-1)+b];
          }
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
  // NOTE: Something is wrong for n = 3 and k = 2, which i am not able to figure out.
  // otherwise seems working

  const auto n = output.size(0); // assuming height and width to be the same
  const auto k = kernel.size(0); // assuming square kernel

  for (int d = 1; d <= 2*n-1; d++) { // Iterating over diagonal index


    // all elements of the dth diagonal computed in parallel
    int threads = d;
    if (d > n) {
      threads = 2*n-d;
    }
    const int blocks = 1; // use multiple blocks if 2d-1 > threads

    AT_DISPATCH_FLOATING_TYPES(input.type(), "cinc_inverse_cuda", ([&] {
      cinc_cuda_inverse_kernel<scalar_t><<<blocks, threads>>>(
          input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          kernel.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(), d);
    }));

    // synchronize all threads
    cudaDeviceSynchronize();

  }

  return {output};
}

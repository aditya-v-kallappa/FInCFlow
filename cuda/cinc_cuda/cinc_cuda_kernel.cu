#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

template <typename scalar_t>
__global__ void cinc_cuda_inverse_kernel(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> kernel,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output/*, int d*/) {
  
  // thread id
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Compute the index from the dth diagonal assigned to the thread
  // int i, j;
  // if (d < n/2) {
  //   i = 2*d - 1 - tid;
  //   j = tid;
  // } else {
  //   j = 2*d - 1 - tid;
  //   i = tid;
  // }

  // compute entry of the output in the diagonal d assigned to this thread
  // output[i][j] = input[i][j]
  // for (int a = k; a >= 0; a--) {
  //    for (int b = k; b >= 0; b--) {
  //         output[i][j] -= kernel[a][b]*output[i-a][j-b];
  //    }
  // }
}

} // namespace

std::vector<torch::Tensor> cinc_cuda_inverse(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor output) {

  // assuming batch size 1 for initial implementation
  // all the tensors above are 2D only

  const auto n = input.size(0); // assuming height and width to be the same
  const auto k = kernel.size(0); // assuming square kernel

  // for (int d = 1; d <= 2n - 1; d++) { // Iterating over diagonal index


  // all elements of the dth diagonal computed in parallel
  const int threads = 1024; // this is the recommended value for # of threads
  const int blocks = (2*d - 1 + threads - 1) / threads; // use multiple blocks if 2d-1 > threads

  AT_DISPATCH_FLOATING_TYPES(input.type(), "cinc_inverse_cuda", ([&] {
    cinc_cuda_inverse_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        kernel.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>()/*, d*/);
  }));

  // synchronize all threads
  // cudaDeviceSynchronize();


  // }

  return {output};
}

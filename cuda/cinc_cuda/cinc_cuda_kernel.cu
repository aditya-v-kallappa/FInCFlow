#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

template <typename scalar_t>
__global__ void cinc_cuda_inverse_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> kernel,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> output) {
  //batch index
  const int n = blockIdx.y;
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;

  // TODO: Compute inverse in a block here

}

} // namespace

std::vector<torch::Tensor> cinc_cuda_inverse(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor output) {

  const auto batch_size = input.size(0);
  const auto state_size = input.size(1);

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "cinc_inverse_cuda", ([&] {
    cinc_cuda_inverse_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        kernel.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        output.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>());
  }));

  return {output};
}

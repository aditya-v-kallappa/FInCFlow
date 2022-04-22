#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> cinc_cuda_inverse_level1(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor output);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> cinc_inverse_level1(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor output) {
  CHECK_INPUT(input);
  CHECK_INPUT(kernel);
  CHECK_INPUT(output);

  return cinc_cuda_inverse_level1(input, kernel, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("inverse", &cinc_inverse_level1, "CINC Inverse level 1 (CUDA)");
}

#include <torch/extension.h>

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {


// Implementation for batch size = 1, channels = 1
template <typename scalar_t>
__global__ void cinc_cuda_inverse_kernel(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits,size_t> kernel,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits,size_t> output,
    int c,  
    int d,
    int relevant_threads)

{

    //scheme 1
    // const auto tid = threadIdx.x; 
    // if (tid >= relevant_threads) return;
    // const auto b = blockIdx.x;   // Current Batch


    const auto b = threadIdx.x;
    const auto tid = blockIdx.x;   

    
    const auto B = output.size(0);  
    const auto C = output.size(1);
    const auto H = output.size(2);
    const auto W = output.size(3);

    // const auto m  = output.size(0); // assuming height and width to be the same
    // const auto n  = output.size(1);
    // NOTE: For some reason kernel.size(0) doesnot work correctly here
    const auto K_H = kernel.size(2);
    const auto K_W = kernel.size(3);
    // Compute the index from the dth diagonal assigned to the thread
    int h, w, n = H;
    // h = tid % m; // batch index encoded in lower indices modulo batchsize
    // tid = (tid - h) / m; // remaining part of tid has the index along the diagonal
    if (d <= n) {
        h = d - 1 - tid;
        w = tid;
    } else {
        w = (d - n)  + tid;
        h = n - 1 - tid;
    }

    // compute entry of the output in the diagonal d assigned to this thread
    output[b][c][h][w] = input[b][c][h][w];
    for (int k_h = 0; k_h < K_H; k_h++) {
        if (h - k_h < 0) break;
        for (int k_w = 0; k_w < K_W; k_w++) {
            if (w - k_w < 0) break;
            for (int k_c = 0; k_c < C; k_c++) {
                if (k_h == 0 && k_w == 0) {
                    if (k_c == c) continue;
                }
                output[b][c][h][w] -= output[b][k_c][h - k_h][w - k_w] * kernel[c][k_c][K_H - k_h - 1][K_W - k_w - 1]; 
            }
        }
    }
}

} // namespace

std::vector<torch::Tensor> cinc_cuda_inverse_level1(
    torch::Tensor input, // B, C, H, W
    torch::Tensor kernel, // C, C, K_H, K_W
    torch::Tensor output)  // B, C, H, W
{
  // assuming batch size 1 for initial implementation
  // all the tensors above are 2D only
    const auto B = output.size(0);
    const auto C = output.size(1);
    const auto H = output.size(2);
    const auto W = output.size(3);

    // printf("%d %d", n,k);
    int n = H < W ? H : W; //samller dimension  -- min(H, W)
    int m = H > W ? H : W; //larger dimension   -- max(H, W)
    // scheme 1
    // dim3 threads(1024, 1, 1);         // Fix the number of threads
    // dim3 blocks(B, 1, 1);             // Fix the number of grids = Batch_Size

    

    for (int d = 1; d <= H + W - 1; d++) { // Iterating over diagonal index
        for (int c = 0; c < C; c++) {
            // const int threads = 1024;
            
            // all elements of the dth diagonal computed in parallel
            int relevant_threads = d;         // Since we have fixed the threads, not all threads are going to be useful
            if (d > n) {
                if (d <= m) {
                    relevant_threads = n;
                } else {  // equivalent to if (d > m) 
                    // threads = 2*n-d;
                    relevant_threads = m + n - d;
                }
            }
            //scheme 2
            dim3 threads(B, 1, 1);
            dim3 blocks(relevant_threads, 1, 1);

            AT_DISPATCH_FLOATING_TYPES(input.type(), "cinc_inverse_cuda", ([&] {
                cinc_cuda_inverse_kernel<scalar_t><<<blocks, threads>>>(
                    input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits,size_t>(),
                    kernel.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits,size_t>(),
                    output.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits,size_t>(),
                    c,
                    d,
                    relevant_threads
                );
            }));

            // synchronize all threads
            cudaDeviceSynchronize();

        }
    }

  return {output};
}

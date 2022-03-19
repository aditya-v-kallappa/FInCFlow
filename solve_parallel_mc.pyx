import numpy as np
cimport numpy as np
cimport cython
import cython.parallel as parallel
from libc.stdio cimport printf


DTYPE = np.float64


ctypedef np.float64_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)

# def inverse_conv(np.ndarray[DTYPE_t, ndim=4] z_np, np.ndarray[DTYPE_t, ndim=4] w_np, int is_upper, int dilation):
#     assert z_np.dtype == DTYPE and w_np.dtype == DTYPE

#     cdef int batchsize = z_np.shape[0]
#     cdef int height = z_np.shape[1]
#     cdef int width = z_np.shape[2]
#     cdef int n_channels = z_np.shape[3]
#     cdef int ksize = w_np.shape[0]
#     cdef int kcenter = (ksize - 1) // 2

#     cdef np.ndarray[DTYPE_t, ndim=4] x_np = np.zeros([batchsize, height, width, n_channels], dtype=DTYPE)

#     cdef int b, j, i, c_out, c_in, k, m, j_, i_, _j, _i, _c_out

#     # Single threaded
#     # for b in range(batchsize):

#     # Multi-threaded. Set max number of threads to avoid mem crash.
#     for b in parallel.prange(batchsize, nogil=True, num_threads=8):

#         # Debug multi-threaded
#         # cdef int thread_id = parallel.threadid()
#         # printf("Thread ID: %d\n", thread_id)

#         for _j in range(height):
#             j = _j if is_upper else height - _j - 1

#             for _i in range(width):
#                 i = _i if is_upper else width - _i - 1

#                 for _c_out in range(n_channels):
#                     c_out = n_channels - _c_out - 1 if is_upper else _c_out

#                     for c_in in range(n_channels):
#                         for k in range(ksize):
#                             for m in range(ksize):
#                                 if k == kcenter and m == kcenter and \
#                                         c_in == c_out:
#                                     continue

#                                 j_ = j + (k - kcenter) * dilation
#                                 i_ = i + (m - kcenter) * dilation

#                                 if not ((j_ >= 0) and (j_ < height)):
#                                     continue

#                                 if not ((i_ >= 0) and (i_ < width)):
#                                     continue

#                                 x_np[b, j, i, c_out] -= w_np[k, m, c_in, c_out] * x_np[b, j_, i_, c_in]

#                     # Compute value for x
#                     x_np[b, j, i, c_out] += z_np[b, j, i, c_out]
#                     x_np[b, j, i, c_out] /= \
#                         w_np[kcenter, kcenter, c_out, c_out]

#     return x_np


def solve_parallel(np.ndarray[DTYPE_t, ndim=4] x, np.ndarray[DTYPE_t, ndim=4] conv_w, k_size):

    # B, C, H, W = x.shape
    cdef int B = x.shape[0]
    cdef int C = x.shape[1]
    cdef int H = x.shape[2]
    cdef int W = x.shape[3]
    # y = x.clone()
    cdef np.ndarray[DTYPE_t, ndim=4] y = x #np.zeros([B, C, H, W], dtype=DTYPE)

    # k_H, k_W = k_size[0], k_size[1]
    cdef int C_out = C
    cdef int C_in = C
    cdef int k_H = conv_w.shape[2] 
    cdef int k_W = conv_w.shape[3]

    cdef int n_steps, n_parallel_op, b, c, i, j, h, w, k_h, k_w, k_c

    if not H % 2 and W % 2:
        n_steps = 2 * W 
    else:
        n_steps = 2 * W - 1
    max_parallel_op = max(H, W)
    for b in range(B):
        for i in range(n_steps):
            # for j in range(max_parallel_op):
            for c in range(C):
                for j in parallel.prange(max_parallel_op, nogil=True, num_threads=30):
                    if j > i:
                        break
                    h, w = j, i - j
                    if h >= H or w >= W:
                        continue 
                    # M_row = h * W + w
                    for k_h in range(k_H):
                        if h - k_h < 0:
                            break
                        for k_w in range(k_W):
                            if w - k_w < 0:
                                break
                            for k_c in range(C):
                                if (k_h == 0 and k_w == 0):
                                    if k_c == c:
                                        continue
                                    if c - k_c < 0:
                                        break
                                y[b, c, h, w] -= y[b, k_c, h - k_h, w - k_w] * \
                                                    conv_w[c, k_c, k_H - k_h - 1, k_W - k_w - 1]
    
    return y
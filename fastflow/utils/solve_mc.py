import torch
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn

torch.random.manual_seed(10)

def solve_parallel(x, conv_w, k_size):
    B, C, H, W = x.shape
    y = x.clone()
    k_H, k_W = k_size[0], k_size[1]
    if not H % 2 and W % 2:
        n_steps = 2 * W 
    else:
        n_steps = 2 * W - 1
    n_parallel_op = min(H, W)
    for b in range(B):
        for i in range(n_steps):
            for c in range(C):
                for j in range(max(H, W)):
                
                    if j > i:
                        break
                    pixel = (j, i - j)
                    h = pixel[0]
                    w = pixel[1]
                    if h >= H or w >= W:
                        continue 
                    M_row = h * W + w
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
                                # M_col = M_row - (k_w + k_h * W)
                                # if h - k_h < 0 or w - k_w < 0 or (k_h == 0 and k_w == 0):
                                #     continue
                                # print(h, w)
                                # y[b, c, h, w] -= y[b, k_c, h - k_h, w - k_w] * M[M_row, M_col]
    
    return y


# def solve(M, x, k_size):
#     B, C, H, W = x.shape
#     y = x.clone()
#     k_H, k_W = k_size[0], k_size[1]
#     for b in range(B):
#         for h in range(H):
#             # if h == 2:
#             #     break
#             for w in range(W):
#                 for c in range(C):
#                     # print(b, c, h, w) # 0 1 2 2
#                     M_row = h * W * C + w * C + c # 2 * 3 * 2 + 2 * 3 + 1 = 19
#                     for k_h in range(k_H):
#                         if h - k_h < 0:
#                             break
#                         for k_w in range(k_W):
#                             if w - k_w < 0:
#                                 break
#                             for k_c in range(C):
#                                 if k_h == 0 and k_w == 0:
#                                     if k_c == c:
#                                         continue
#                                     if c - k_c < 0:
#                                         break
                                
#                                 # M_col = M_row - (k_w * C + k_h * W * C) + k_c
#                                 # M_col = k_c + k_W * C + k_h * W * C
#                                 M_col = (h - k_h) * W * C + (w - k_w) * C + k_c#(c - k_c) % C
#                                 # print((b, c, h, w), (k_h, k_w, k_c), (M_row, M_col))
#                                 y[b, c, h, w] -= y[b, k_c, h - k_h, w - k_w] * M[M_row, M_col]
    

#     return y


def solve(x, conv_w, k_size):
    B, C, H, W = x.shape
    y = x.clone()
    k_H, k_W = k_size[0], k_size[1]
    for b in range(B):
        for h in range(H):
            # if h == 2:
            #     break
            for w in range(W):
                for c in range(C):
                    for k_h in range(k_H):
                        if h - k_h < 0:
                            break
                        for k_w in range(k_W):
                            if w - k_w < 0:
                                break
                            for k_c in range(C):
                                if k_h == 0 and k_w == 0:
                                    if k_c == c:
                                        continue
                                    if c - k_c < 0:
                                        break
                                y[b, c, h, w] -= y[b, k_c, h - k_h, w - k_w] * \
                                                    conv_w[c, k_c, k_H - k_h - 1, k_W - k_w - 1]
    

    return y

# def construct_matrix(x, conv_w):
#     B, C, H, W = x.shape
#     C_out, C_in, k_h, k_w = conv_w.shape
#     M = torch.zeros(size=(C * H * W, C * H * W), dtype=conv_w.dtype)
    
#     # flattened_w = torch.ravel(conv_w)[::-1]

#     flattened_w = conv_w.flip(0, 1).permute((2, 3, 0, 1)).flatten().flip(0)
#     # print(flattened_w)
#     submatrix1 = torch.zeros(size=(W, C, C)) 
#     submatrix = torch.zeros(size=(H, W * C, W * C))
#     # print("Submatrix1:", submatrix1.shape)
#     # print("Submatrix :", submatrix.shape)
#     for i in range(k_h):
#         start = i * C * C * k_w
#         for j in range(k_w):#(W):
#             for c in range(C):
#                 for k in range(C):
#                     submatrix1[j, c, k] = flattened_w[start + C * C * j + c * C + k]

#         for w in range(W):
#             for j in range(w + 1):
#                 submatrix[i, C * w:C*(w+1), C*j:C*(j+1)] = submatrix1[w - j]
#     for i in range(H):
#         for j in range(i + 1):
#             M[W * C * i:W * C * (i + 1), W * C * j:W * C * (j + 1)] = submatrix[i - j]
#     return submatrix, M

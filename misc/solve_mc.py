import torch
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

#torch.random.manual_seed(10)

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

def construct_matrix(x, conv_w):
    B, C, H, W = x.shape
    C_out, C_in, k_h, k_w = conv_w.shape
    M = torch.zeros(size=(C * H * W, C * H * W), dtype=conv_w.dtype)
    
    # flattened_w = torch.ravel(conv_w)[::-1]

    flattened_w = conv_w.flip(0, 1).permute((2, 3, 0, 1)).flatten().flip(0)
    # print(flattened_w)
    submatrix1 = torch.zeros(size=(W, C, C)) 
    submatrix = torch.zeros(size=(H, W * C, W * C))
    # print("Submatrix1:", submatrix1.shape)
    # print("Submatrix :", submatrix.shape)
    for i in range(k_h):
        start = i * C * C * k_w
        for j in range(k_w):#(W):
            for c in range(C):
                for k in range(C):
                    submatrix1[j, c, k] = flattened_w[start + C * C * j + c * C + k]

        for w in range(W):
            for j in range(w + 1):
                submatrix[i, C * w:C*(w+1), C*j:C*(j+1)] = submatrix1[w - j]
    for i in range(H):
        for j in range(i + 1):
            M[W * C * i:W * C * (i + 1), W * C * j:W * C * (j + 1)] = submatrix[i - j]
    return submatrix, M


def main():
    B = 1
    C = 3
    k_size = (3, 3)
    img_size = (5, 5)
    H, W = img_size[0], img_size[1]
    conv_w = torch.randn(k_size[0] * k_size[1] * C * C).reshape(C, C, k_size[0], k_size[1])
    for c_out in range(C):
        conv_w[c_out, c_out, -1, -1] = 1.0
        conv_w[c_out, c_out+1:, -1, -1] = 0.0
    input = torch.randn(H * W * C).reshape(B, C, H, W).to(torch.float32)
    m = nn.ConstantPad2d((k_size[0] - 1, 0, k_size[1] - 1, 0), 0)
    x = torch.nn.functional.conv2d(m(input), conv_w)


    # _, M = construct_matrix(x, conv_w)
    _yp = solve_parallel(x, conv_w, k_size)


    # print("Solve parallel:\n", _yp)
    # print("Correct\n", input)
    print("Error:", torch.sqrt(((input - _yp)**2).mean()))


    # correct_answer = M @ input.permute(0, 2, 3, 1).squeeze().flatten()
    # print(correct_answer.reshape(B, H, W, C).permute(0, 3, 1, 2))
    # print(torch.sqrt((x - correct_answer.reshape(B, H, W, C).permute(0, 3, 1, 2))**2).mean())

if __name__ == '__main__':

    # main()
    B = 1
    C = 2
    k_size = (3, 3)
    img_size = (4, 4)
    H, W = img_size[0], img_size[1]
    conv_w = torch.rand(k_size[0] * k_size[1] * C * C).reshape(C, C, k_size[0], k_size[1])
    # conv_w[conv_w >= 0.95] = 0.85
    # conv_w[conv_w <= 0.05] = 0.10
    for c_out in range(C):
        conv_w[c_out, c_out, -1, -1] = 1.0
        conv_w[c_out, c_out+1:, -1, -1] = 0.0
    x = torch.randn(H * W * C).reshape(B, C, H, W).to(torch.float32)

    _, M = construct_matrix(x, conv_w)

    # # for i in range(len(M)):
    # #     print(M[i, i])

    top = cm.get_cmap('hot_r', 128)
    bottom = cm.get_cmap('Blues', 128)

    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                        bottom(np.linspace(0, 1, 128))))
    newcmp = ListedColormap(newcolors, name='hotBlues')

    # x_ticks = [h*W*C for h in range(H+1)]
    # y_ticks = [w*W*C for w in range(H+1)]
    # fig, ax = plt.subplots(figsize=(15, 15))
    # ax.matshow(M.numpy(), cmap=newcmp, extent=[0, H*W*C, 0, H*W*C])
    # ax.set_xticks(x_ticks)
    # ax.set_yticks(y_ticks)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # plt.grid(color='black', linewidth=1)
    # plt.savefig("matrix.png", bbox_inches='tight')

    # plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(5, 10))
    x_ticks2 = [h*W*C for h in range(H+1)]
    y_ticks2 = [w*W*C for w in range(H+1)]
    # fig, ax = plt.subplots(figsize=(15, 15))
    ax[1].matshow(M.numpy(), cmap=newcmp, extent=[0, H*W*C, 0, H*W*C])
    ax[1].set_xticks(x_ticks2)
    ax[1].set_yticks(y_ticks2)
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].grid(color='black', linewidth=1)
    
    H, W = 6, 6
    M2 = np.ones(shape=(H, W))
    value = 1
    alpha = 0.9
    m = max(H, W)
    n = min(H, W)

    for d in range(H + W - 1):
        if (d < n):
            for i in range(d + 1):
                M2[i, d - i] =  value
        else:
            if d < m:
                for i in range(n):
                    M2[i, d - i] = value
            else:
                temp = d - m + 1
                for i in range(temp, m):
                    M2[i, d - i] = value
        value = value * alpha
    M2[-2, -1] = M[-1, -2] = 0.2
    M2[-1, -1] = 0

    
    ax[0].matshow(M2, cmap='tab20_r', extent=[0, M2.shape[0], 0, M2.shape[1]])
    ax[0].set_xticks(range(M2.shape[0]))
    ax[0].set_yticks(range(M2.shape[1]))
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[0].grid(color='black', linewidth=1)

    plt.savefig('2.png', bbox_inches='tight')
    plt.plot()
    
    


import torch
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn

def solve_parallel(M, x, k_size):
    B, C, H, W = x.shape
    y = x.clone()
    k_H, k_W = k_size[0], k_size[1]
    if not H % 2 and W % 2:
        n_steps = 2 * W 
    else:
        n_steps = 2 * W - 1
    n_parallel_op = min(H, W)
    for b in range(B):
        for c in range(C):
            for i in range(n_steps):
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
                            if (k_h == 0 and k_w == 0):
                                continue
                            if w - k_w < 0:
                                break
                            M_col = M_row - (k_w + k_h * W)
                            if h - k_h < 0 or w - k_w < 0 or (k_h == 0 and k_w == 0):
                                continue
                            # print(h, w)
                            y[b, c, h, w] -= y[b, c, h - k_h, w - k_w] * M[M_row, M_col]
    
    return y


def _solve(M, x, k_size):
    B, C, H, W = x.shape
    y = x.clone()
    k_H, k_W = k_size[0], k_size[1]
    for b in range(B):
        for c in range(C):
            for i in range(H):
                for j in range(W):
                    M_row = i * W + j
                    for k_h in range(k_H):
                        if i - k_h < 0:
                            break
                        for k_w in range(k_W):
                            if (k_h == 0 and k_w == 0):
                                continue
                            if j - k_w < 0:
                                break
                            M_col = M_row - (k_w + k_h * W)
                            y[b, c, i, j] -= y[b, c, i - k_h, j - k_w] * M[M_row, M_col]
    

    return y

def construct_matrix(x, conv_w):
    B, C, H, W = x.shape
    C_out, C_in, k_h, k_w = conv_w.shape
    M = torch.zeros(size=(C * H * W, C * H * W), dtype=conv_w.dtype)
    
    # flattened_w = torch.ravel(conv_w)[::-1]
    for c in range(C):
        flattened_w = torch.flatten(conv_w[c]).flip(0)
        # print("Flatted W:\n", flattened_w)
        submatrix = torch.zeros(size=(H, W, W))

        # for i in range(C * H * W):
        #     for j in range(C * H * W):
        #         M[i, j] = 
        # for c in range(C):
        for i in range(k_h):#(H):
            for j in range(W):
                for k in range(j+1):
                    if j - k >= k_w: #or i >= k_h:
                        continue
                    submatrix[i, j, k] = flattened_w[k_w * i + j - k]

        # print("Submatrix:\n", submatrix)

        for i in range(H):
            for j in range(i + 1):
                # start_w = i * (W + 1) 
                M[c * H * W + W * i:c * H * W + W*(i+1), c * H * W + W*j:c * H * W + W*(j+1)] = submatrix[i - j]
    
    # print("M:\n", M)
    return submatrix, M

B = 1
C = 2
k_size = (3, 3)
img_size = (4, 4)
H, W = img_size[0], img_size[1]
conv_w = torch.randn(k_size[0] * k_size[1] * C * C).reshape(C, C, k_size[0], k_size[1])
conv_w[:, -1, -1, -1] = 1.0
x = torch.randn(H * W * C).reshape(B, C, H, W).to(torch.float32)
y = torch.ones_like(x)

_, M = construct_matrix(x, conv_w)

# _x = torch.flatten(x, 0, -1)
# _y = torch.flatten(y, 0, -1)

# _y[0] = _x[0]

x_ticks = [h*W for h in range(H+1)]
y_ticks = [w*W for w in range(H+1)]
fig, ax = plt.subplots(figsize=(15, 15))
ax.matshow(M.numpy(), cmap='seismic')
ax.set_xticks(y_ticks)
ax.set_yticks(x_ticks)
plt.savefig("matrix.png")
plt.show()

# inv_M = torch.tensor(np.linalg.inv(M.numpy()))
# _x = x.squeeze().flatten(0, -1)
# # print(_x.shape)
# y = (inv_M @ _x).reshape(x.shape)

# _y = _solve(M, x, k_size)
# # __y = solve_parallel(M, x, k_size)
# print("Error", torch.sqrt((y - _y)**2).mean())
# print("Error 2 ", torch.sqrt((y - __y)**2).mean())
# print("Error 3 ", torch.sqrt((_y - __y)**2).mean())


# print("y:\n", y)
# print("_y:\n", _y)
# print("__y:\n", __y)
# print("M:\n", M)

# answer = (M @ y.squeeze().flatten(0, -1)).reshape(y.shape)
# print("Inverted Answer:\n", answer)

# m = nn.ConstantPad2d((2, 0, 2, 0), 0)
# print("Padded y:\n", m(y))
# print("Weights:\n", conv_w)
# conv_answer = torch.nn.functional.conv2d(m(y), conv_w)
# print("Conv answer:\n", conv_answer)
# print(x)
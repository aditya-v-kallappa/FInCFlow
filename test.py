import solve_parallel_mc
from solve_mc import solve_parallel
import time
import torch
import torch.nn as nn
import numpy as np

B = 1
C = 3
k_size = (3, 3)
img_size = (3, 3)
H, W = img_size[0], img_size[1]
conv_w = torch.randn(k_size[0] * k_size[1] * C * C).reshape(C, C, k_size[0], k_size[1])
for c_out in range(C):
    conv_w[c_out, c_out, -1, -1] = 1.0
    conv_w[c_out, c_out+1:, -1, -1] = 0.0
input = torch.randn(H * W * C).reshape(B, C, H, W).to(torch.float32)
m = nn.ConstantPad2d((k_size[0] - 1, 0, k_size[1] - 1, 0), 0)
x = torch.nn.functional.conv2d(m(input), conv_w)

x_np = x.detach().numpy()
x_np = np.asarray(x_np, dtype=np.float64)
conv_w_np = conv_w.detach().numpy()
conv_w_np = np.asarray(conv_w_np, dtype=np.float64)
start_no_parallel = time.time()
y = solve_parallel(x, conv_w, k_size)
end_no_parallel = time.time()

start_parallel = time.time()
yp = solve_parallel_mc.solve_parallel(x_np, conv_w_np, k_size)
end_parallel = time.time()

print("No parallel time:", end_no_parallel - start_no_parallel)
print("Parallel time:", end_parallel - start_parallel)
# error = torch.nn.functional.mse_loss(torch.tensor(y), torch.tensor(yp))
# print("Error:", error)

print(input)
print(y)
print(np.round(yp, 4))
import torch

from fastflow import FastFlowUnit

x = torch.rand((10, 4, 3, 1))
model = FastFlowUnit(in_channels=x.shape[1], out_channels=x.shape[1], kernel_size=3)
print(model)

model.eval()
out, _ = model.inverse(x)

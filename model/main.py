import torch

from fastflow import FastFlowUnit

x = torch.rand((1, 8, 10, 10))
model = FastFlowUnit(in_channels=x.shape[1], out_channels=x.shape[1], kernel_size=3)
print(model)

model.eval()
out, _ = model(x)
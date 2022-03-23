import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Model2
from matplotlib import pyplot as plt
import numpy as np

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model = Model2(20).to(device)
model.load_state_dict(torch.load('./saved_model_normalized.pt'))
model.eval()
B = 25
fig, ax = plt.subplots(5, 5)

x = torch.randn(B*28*28).reshape(B, 1, 28, 28).to(device)

images, _ = model.reverse(x)
clipped_images = torch.clip(images, 0, 1) * 255
normalized_images = images / torch.max(torch.abs(images))

for i in range(5):
    for j in range(5):
        v = j + i * 5
        clipped_image = clipped_images[v].squeeze().numpy()
        clipped_image = np.array(clipped_image, dtype=np.uint8)
        ax[i, j].imshow(clipped_image, cmap='gray')
        


# normalized_images = normalized_images.squeeze().numpy()

# plt.imshow(clipped_images, cmap='gray')
plt.show()
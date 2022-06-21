import torch

def preprocess(img, n_bits, noise=None):
    n_bins = 2. ** n_bits
    # rescale to 255
    img = img.mul(255)
    if n_bits < 8:
        img = torch.floor(img.div(256. / n_bins))

    if noise is not None:
        # [batch, nsamples, channels, H, W]
        img = img.unsqueeze(1) + noise
    # normalize
    img = img.div(n_bins)
    # img = (img - 0.5).div(0.5)
    return img


def postprocess(img, n_bits):
    n_bins = 2. ** n_bits
    # re-normalize
    img = img.mul(0.5) + 0.5
    img = img.mul(n_bins)
    # scale
    img = torch.floor(img) * (256. / n_bins)
    img = img.clamp(0, 255).div(255)
    return img
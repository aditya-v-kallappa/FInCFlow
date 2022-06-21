import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.fastflow_inverse.solve_parallel_mc import solve_parallel
from utils.solve_mc import solve
from .flowlayer import FlowLayer
import numpy as np

from torch.utils.cpp_extension import load

# cinc_cuda_level1 = load(
#     'cinc_cuda_level1', ['utils/fastflow_cuda_inverse/cinc_cuda_level1.cpp', 'utils/fastflow_cuda_inverse/cinc_cuda_kernel_level1.cu'], verbose=True)


# def _reverse(x, conv_w=None):
#     """
#     TO be implemented in CUDA
#     """
#     return x

class PaddedConv2d(FlowLayer):
    """
    Conv2d with specific padded order
    TL -> top-left padding
    TR -> top-right
    BL -> bottom-left
    BR -> bottom-right
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=False, order='TL'):
        super().__init__()
        # assert len(stride) == 2
        assert len(kernel_size) == 2
        assert order in {'TL', 'TR', 'BL', 'BR'}, 'unknown order: {}'.format(order)
       
        self.kernel_size = kernel_size
        self.order = order

        K_H, K_W = kernel_size[0], kernel_size[1]

        if order == 'TL': # top & left padding
            # left, right, top, bottom
            self.pad = (K_W - 1, 0, K_H - 1, 0)
            
        elif order == 'TR': # top & right padding
            # left, right, top, bottom
            self.pad = (0, K_W - 1, K_H - 1, 0)
            
        elif order == 'BL': # bottom & left padding
            # left, right, top, bottom
            self.pad = (K_W - 1, 0, 0, K_H - 1)
            
        elif order == 'BR': # bottom & right pad
            # left, right, top, bottom
            self.pad = (0, K_W - 1, 0, K_H - 1)
            
        else:
            raise ValueError('unknown order: {}'.format(order))

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.conv.weight.data, mean=0.0, std=0.05)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
        self.mask = self.get_mask()
        for c_out in range(self.conv.weight.data.shape[0]):
            self.conv.weight.data[c_out, c_out, -1, -1] = 1.0
            self.conv.weight.data[c_out, c_out+1:, -1, -1] = 0.0
        
        if self.order == 'TR':
            self.conv.weight.data = torch.flip(self.conv.weight.data, [3])
        
        elif self.order == 'BL':
            self.conv.weight.data = torch.flip(self.conv.weight.data, [2])
        
        elif self.order == 'BR':
            self.conv.weight.data = torch.flip(self.conv.weight.data, [2, 3])

    def get_mask(self):
        mask = torch.ones_like(self.conv.weight)
        for c_out in range(self.conv.weight.data.shape[0]):
            mask[c_out, c_out, -1, -1] = 0.0
            mask[c_out, c_out+1:, -1, -1] = 0.0
        
        if self.order == 'TR':
            mask = torch.flip(mask, [3])
        
        elif self.order == 'BL':
            mask = torch.flip(mask, [2])
        
        elif self.order == 'BR':
            mask = torch.flip(mask, [2, 3])
        
        return mask

    def reset_gradients(self):
        self.conv.weight.grad = self.conv.weight.grad * self.mask.to(self.conv.weight.grad.device)

    
    def forward(self, x, context=None, compute_expensive=None):
        x = F.pad(x, self.pad)
        # self.conv = self.conv.to(x.device)
        out = self.conv(x)
        logdet = 0.0
        return out, logdet

    def reverse(self, x, context=None, compute_expensive=None):
        # return self.reverse_cuda(x)
        return self.reverse_cython(x)

    def reverse_cython(self, x):
        input = x
        C = x.shape[1]
        if self.conv.bias is not None:
            x = x - self.conv.bias.reshape(-1, C, 1, 1)
        if self.order == 'TR':
            x = torch.flip(x, [3])
            # y = solve(x, torch.flip(self.conv.weight.data, [3]), self.kernel_size)
            y = solve_parallel(
                np.asarray(x.detach().cpu().numpy(), dtype=np.float64),
                np.asarray(torch.flip(self.conv.weight.data, [3]).detach().cpu().numpy(), dtype=np.float64),
                self.kernel_size
            )    
            y = torch.from_numpy(y).to(torch.float32).to(x.device)
            y = torch.flip(y, [3])
        
        elif self.order == 'BL':
            x = torch.flip(x, [2])
            # y = solve(x, torch.flip(self.conv.weight.data, [2]), self.kernel_size)
            y = solve_parallel(
                np.asarray(x.detach().cpu().numpy(), dtype=np.float64),
                np.asarray(torch.flip(self.conv.weight.data, [2]).detach().cpu().numpy(), dtype=np.float64),
                self.kernel_size
            ) 
            y = torch.from_numpy(y).to(torch.float32).to(x.device)
            y = torch.flip(y, [2])
        
        elif self.order == 'BR':
            x = torch.flip(x, [2, 3])
            # y = solve(x, torch.flip(self.conv.weight.data, [2, 3]), self.kernel_size)
            y = solve_parallel(
                np.asarray(x.detach().cpu().numpy(), dtype=np.float64),
                np.asarray(torch.flip(self.conv.weight.data, [2, 3]).detach().cpu().numpy(), dtype=np.float64),
                self.kernel_size
            ) 
            y = torch.from_numpy(y).to(torch.float32).to(x.device)
            y = torch.flip(y, [2, 3])
        else:
            # y = solve(x, self.conv.weight.data, self.kernel_size)
            y = solve_parallel(
                np.asarray(x.detach().cpu().numpy(), dtype=np.float64),
                np.asarray(self.conv.weight.data.detach().cpu().numpy(), dtype=np.float64),
                self.kernel_size
            ) 
            y = torch.from_numpy(y).to(torch.float32).to(x.device)

        # conv_w_np = self.conv.weight.data.cpu().detach().numpy()
        # x_np = x.cpu().detach().numpy()
        # ys = solve_seq.solve(x_np, conv_w_np, self.kernel_size)
        logdet = 0
        return y, logdet

    def reverse_python(self, x):
        input = x
        C = x.shape[1]
        if self.conv.bias is not None:
            x = x - self.conv.bias.reshape(-1, C, 1, 1)
        if self.order == 'TR':
            x = torch.flip(x, [3])
            y = solve(x, torch.flip(self.conv.weight.data, [3]), self.kernel_size)
            y = torch.flip(y, [3])
        
        elif self.order == 'BL':
            x = torch.flip(x, [2])
            y = solve(x, torch.flip(self.conv.weight.data, [2]), self.kernel_size)
            y = torch.flip(y, [2])
        
        elif self.order == 'BR':
            x = torch.flip(x, [2, 3])
            y = solve(x, torch.flip(self.conv.weight.data, [2, 3]), self.kernel_size)
            y = torch.flip(y, [2, 3])
        else:
            y = solve(x, self.conv.weight.data, self.kernel_size)
        

        logdet = 0
        return y, logdet

    def reverse_cuda(self, x):
        input = x
        y = torch.zeros_like(x).to(x.device)
        C = x.shape[1]
        if self.conv.bias is not None:
            x = x - self.conv.bias.reshape(-1, C, 1, 1)
        if self.order == 'TR':
            x = torch.flip(x, [3])
            kernel = torch.flip(self.conv.weight.data, [3])
            y = cinc_cuda_level1.inverse(x, kernel, y)[0]
            y = torch.flip(y, [3])
        
        elif self.order == 'BL':
            x = torch.flip(x, [2])
            kernel = torch.flip(self.conv.weight.data, [2])
            y = cinc_cuda_level1.inverse(x, kernel, y)[0]
            y = torch.flip(y, [2])
        
        elif self.order == 'BR':
            x = torch.flip(x, [2, 3])
            kernel = torch.flip(self.conv.weight.data, [2, 3])
            y = cinc_cuda_level1.inverse(x, kernel, y)[0]
            y = torch.flip(y, [2, 3])
        else:
            kernel = self.conv.weight.data
            y = cinc_cuda_level1.inverse(x, kernel, y)[0]
        
        return y, 0.0

    def logdet(self, x, context=None):
        return 0.0

    
    # def init(self, x, init_scale=1.0):
    #     with torch.no_grad():
    #         # [batch, n_channels, H, W]
    #         out = self(x)
    #         n_channels = out.size(1)
    #         out = out.transpose(0, 1).contiguous().view(n_channels, -1)
    #         # [n_channels]
    #         mean = out.mean(dim=1)
    #         std = out.std(dim=1)
    #         inv_stdv = init_scale / (std + 1e-6)

    #         self.conv.weight_g.mul_(inv_stdv.view(n_channels, 1, 1, 1))
    #         if self.conv.bias is not None:
    #             self.conv.bias.add_(-mean).mul_(inv_stdv)
    #         return self(x)

    

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(Conv1x1, self).__init__()
        assert in_channels == out_channels, "Input and Output are not same"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if kernel_size == 1 else kernel_size
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, padding=0, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.orthogonal_(self.weight)
        nn.init.orthogonal_(self.conv.weight.data)
    
    def forward(self, x):
        B, C, H, W = x.size()
        out = self.conv(x)
        _, logdet = torch.slogdet(self.conv.weight.data.squeeze())
        logdet *= H * W
        return out, logdet 


    def reverse(self, x):
        B, C, H, W = x.size()
        bias = 0.0 if self.conv.bias is None else self.conv.bias
        bias = bias.view(1, -1, 1, 1)
        weight_inv = self.conv.weight.data.squeeze().reverse().view(self.in_channels, self.out_channels, 1, 1)
        out = F.conv2d(torch.sub(x, bias), weight_inv)
        _, logdet = torch.slogdet(weight_inv)
        logdet *= H * W
        return out, logdet

import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils.paddedconvbackward.solve_seq 
from utils.solve_mc import solve
from .flowlayer import FlowLayer

# def _reverse(x, conv_w=None):
#     """
#     TO be implemented in CUDA
#     """
#     return x

class PaddedConv2d(FlowLayer):
    """
    Conv2d with shift operation.
    A -> top
    B -> bottom
    C -> left
    D -> right
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, order='TL'):
        super().__init__()
        # assert len(stride) == 2
        assert len(kernel_size) == 2
        assert order in {'TL', 'TR', 'BL', 'BR'}, 'unknown order: {}'.format(order)
        if order in {'TL', 'TR'}:
            assert kernel_size[1] % 2 == 1, 'kernel width cannot be even number: {}'.format(kernel_size)
        else:
            assert kernel_size[0] % 2 == 1, 'kernel height cannot be even number: {}'.format(kernel_size)
        
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

        

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
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
        self.conv.weight.grad *= self.conv.weight.grad * self.mask.to(self.conv.weight.grad.device)

    
    def forward(self, x, context=None, compute_expensive=None):
        x = F.pad(x, self.pad)
        out = self.conv(x)
        logdet = 0.0
        return out, logdet

    def reverse(self, x, context=None, compute_expensive=None):
        if self.order == 'TR':
            x = torch.flip(x, [3])
        
        elif self.order == 'BL':
            x = torch.flip(x, [2])
        
        elif self.order == 'BR':
            x = torch.flip(x, [2, 3])
        
        y = solve(x, self.conv.weight.data, self.kernel_size)
        # conv_w_np = self.conv.weight.data.cpu().detach().numpy()
        # x_np = x.cpu().detach().numpy()
        # ys = solve_seq.solve(x_np, conv_w_np, self.kernel_size)
        logdet = 0
        return y, logdet

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


# class FFBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, hidden_channels, order):
#         super(FFBlock, self).__init__()
#         self.conv = PaddedConv2d(in_channels, hidden_channels, kernel_size, order=order, bias=True)
#         # self.conv1x1 = Conv2dWeightNorm(hidden_channels, out_channels, kernel_size=1, bias=True)
#         # self.activation = nn.ELU(inplace=True)

#     def forward(self, x, s=None, shifted=True):
#         c = self.conv(x, shifted=shifted)
#         # if s is not None:
#         #     c = c + s
#         # c = self.conv1x1(self.activation(c))
#         return c

#     def init(self, x, s=None, init_scale=1.0):
#         c = self.conv.init(x, init_scale=init_scale)
#         # if s is not None:
#         #     c = c + s
#         # c = self.conv1x1.init(self.activation(c), init_scale=0.0)
#         return c




# class FastFlow(Flow):
#     """
#     Masked Convolutional Flow
#     """

#     def __init__(self, in_channels, kernel_size, hidden_channels=None, s_channels=None, order='A', scale=True, reverse=False):
#         super(FastFlow, self).__init__(reverse)
#         self.in_channels = in_channels
#         self.scale = scale
#         if hidden_channels is None:
#             if in_channels <= 96:
#                 hidden_channels = 4 * in_channels
#             else:
#                 hidden_channels = min(2 * in_channels, 512)
#         out_channels = in_channels
#         if scale:
#             out_channels = out_channels * 2
#         self.kernel_size = kernel_size
#         self.order = order
#         self.net = FFBlock(in_channels, out_channels, kernel_size, hidden_channels, order)
#         if s_channels is None or s_channels <= 0:
#             self.s_conv = None
#         else:
#             self.s_conv = Conv2dWeightNorm(s_channels, hidden_channels, (3, 3), bias=True, padding=1)

#     def calc_mu_and_scale(self, x: torch.Tensor, s=None, shifted=True):
#         mu = self.net(x, s=s, shifted=shifted)
#         scale = None
#         if self.scale:
#             mu, log_scale = mu.chunk(2, dim=1)
#             scale = log_scale.add_(2.).sigmoid_()
#         return mu, scale

#     def init_net(self, x, s=None, init_scale=1.0):
#         mu = self.net.init(x, s=s, init_scale=init_scale)
#         scale = None
#         if self.scale:
#             mu, log_scale = mu.chunk(2, dim=1)
#             scale = log_scale.add_(2.).sigmoid_()
#         return mu, scale

#     @overrides
#     def forward(self, x: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Args:
#             x: Tensor
#                 x tensor [batch, in_channels, H, W]
#             s: Tensor
#                 conditional x (default: None)

#         Returns: out: Tensor , logdet: Tensor
#             out: [batch, in_channels, H, W], the output of the flow
#             logdet: [batch], the log determinant of :math:`\partial output / \partial x`
#         """
#         if self.s_conv is not None:
#             s = self.s_conv(s)
#         mu, scale = self.calc_mu_and_scale(x, s=s)
#         out = x
#         if self.scale:
#             out = out.mul(scale)
#             logdet = scale.log().view(mu.size(0), -1).sum(dim=1)
#         else:
#             logdet = mu.new_zeros(mu.size(0))
#         out = out + mu
#         return out, logdet

#     def backward_height(self, x: torch.Tensor, s=None, reverse=False) -> torch.Tensor:
#         batch, channels, H, W = x.size()

#         kH, kW = self.kernel_size
#         cW = kW // 2
#         out = x.new_zeros(batch, channels, H + kH, W + 2 * cW)

#         itr = reversed(range(H)) if reverse else range(H)
#         for h in itr:
#             curr_h = h if reverse else h + kH
#             s_h = h + 1 if reverse else h
#             t_h = h + kH + 1 if reverse else h + kH
#             # [batch, channels, kH, width+2*cW]
#             out_curr = out[:, :, s_h:t_h]
#             s_curr = None if s is None else s[:, :, h:h + 1]
#             # [batch, channels, width]
#             in_curr = x[:, :, h]

#             # [batch, channels, 1, width]
#             mu, scale = self.calc_mu_and_scale(out_curr, s=s_curr, shifted=False)
#             # [batch, channels, width]
#             new_out = in_curr - mu.squeeze(2)
#             if self.scale:
#                 new_out = new_out.div(scale.squeeze(2) + 1e-12)
#             out[:, :, curr_h, cW:W + cW] = new_out

#         out = out[:, :, :H, cW:cW + W] if reverse else out[:, :, kH:, cW:cW + W]
#         return out

#     def backward_width(self, x: torch.Tensor, s=None, reverse=False) -> torch.Tensor:
#         batch, channels, H, W = x.size()

#         kH, kW = self.kernel_size
#         cH = kH // 2
#         out = x.new_zeros(batch, channels, H + 2 * cH, W + kW)

#         itr = reversed(range(W)) if reverse else range(W)
#         for w in itr:
#             curr_w = w if reverse else w + kW
#             s_w = w + 1 if reverse else w
#             t_w = w + kW + 1 if reverse else w + kW
#             # [batch, channels, height+2*cH, kW]
#             out_curr = out[:, :, :, s_w:t_w]
#             s_curr = None if s is None else s[:, :, :, w:w + 1]
#             # [batch, channels, height]
#             in_curr = x[:, :, :, w]

#             # [batch, channels, height, 1]
#             mu, scale = self.calc_mu_and_scale(out_curr, s=s_curr, shifted=False)
#             # [batch, channels, height]
#             new_out = in_curr - mu.squeeze(3)
#             if self.scale:
#                 new_out = new_out.div(scale.squeeze(3) + 1e-12)
#             out[:, :, cH:H + cH, curr_w] = new_out

#         out = out[:, :, cH:cH + H, :W] if reverse else out[:, :, cH:cH + H, kW:]
#         return out

#     @overrides
#     def backward(self, x: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Args:
#             x: Tensor
#                 x tensor [batch, in_channels, H, W]
#             s: Tensor
#                 conditional x (default: None)

#         Returns: out: Tensor , logdet: Tensor
#             out: [batch, in_channels, H, W], the output of the flow
#             logdet: [batch], the log determinant of :math:`\partial output / \partial x`
#         """
#         if self.s_conv is not None:
#             ss = self.s_conv(s)
#         else:
#             ss = s
#         if self.order == 'A':
#             out = self.backward_height(x, s=ss, reverse=False)
#         elif self.order == 'B':
#             out = self.backward_height(x, s=ss, reverse=True)
#         elif self.order == 'C':
#             out = self.backward_width(x, s=ss, reverse=False)
#         else:
#             out = self.backward_width(x, s=ss, reverse=True)
#         _, logdet = self.forward(out, s=s)
#         return out, logdet.mul(-1.0)

#     @overrides
#     def init(self, data, s=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
#         if self.s_conv is not None:
#             s = self.s_conv.init(s, init_scale=init_scale)
#         mu, scale = self.init_net(data, s=s, init_scale=init_scale)
#         out = data
#         if self.scale:
#             out = out.mul(scale)
#             logdet = scale.log().view(mu.size(0), -1).sum(dim=1)
#         else:
#             logdet = mu.new_zeros(mu.size(0))
#         out = out + mu
#         return out, logdet

#     # @classmethod
#     # def from_params(cls, params: Dict) -> "MaskedConvFlow":
#     #     return MaskedConvFlow(**params)


# class Conv2dWeightNorm(nn.Module):
#     """
#     Conv2d with weight normalization
#     """

#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True):
#         super(Conv2dWeightNorm, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
#                               padding=padding, dilation=dilation, groups=groups, bias=bias)
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.normal_(self.conv.weight, mean=0.0, std=0.05)
#         if self.conv.bias is not None:
#             nn.init.constant_(self.conv.bias, 0)
#         self.conv = nn.utils.weight_norm(self.conv)

#     # def init(self, x, init_scale=1.0):
#     #     with torch.no_grad():
#     #         # [batch, n_channels, H, W]
#     #         out = self(x)
#     #         n_channels = out.size(1)
#     #         out = out.transpose(0, 1).contiguous().view(n_channels, -1)
#     #         # [n_channels]
#     #         mean = out.mean(dim=1)
#     #         std = out.std(dim=1)
#     #         inv_stdv = init_scale / (std + 1e-6)

#     #         self.conv.weight_g.mul_(inv_stdv.view(n_channels, 1, 1, 1))
#     #         if self.conv.bias is not None:
#     #             self.conv.bias.add_(-mean).mul_(inv_stdv)
#     #         return self(x)

        
#     def forward(self, x):
#         return self.conv(x)



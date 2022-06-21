__author__ = 'aditya'

import torch
import torch.nn as nn

from layers.conv import PaddedConv2d, Conv1x1


class CINCFlowUnit(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CINCFlowUnit, self).__init__()
        if isinstance(kernel_size, int) or len(kernel_size) == 1:
            kernel_size = (kernel_size, kernel_size)
        # assert in_channels % 4 == 0, "Input channels have to be a multiple of 4" 

        out_channels = in_channels
        
        
        self.conv_tl = PaddedConv2d(out_channels, out_channels, kernel_size, order='TL')#, bias=True)
        # self.conv_tr = PaddedConv2d(out_channels, out_channels, kernel_size, order='TR')#, bias=True)
        # self.conv_bl = PaddedConv2d(out_channels, out_channels, kernel_size, order='BL')#, bias=True)
        # self.conv_br = PaddedConv2d(out_channels, out_channels, kernel_size, order='BR')#, bias=True)

    
    def forward(self, x, context=None):

        logdet_accum = 0.0

        out, logdet = self.conv_tl.forward(x)
        logdet_accum += logdet
        
    
        return out, logdet_accum

    def reverse(self, x, context=None):
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
            return M

        # B, C, H, W = x.shape
        # M = construct_matrix(x, self.conv_tl.conv.conv.data).to(self.conv_tl.conv.conv.data.device)
        # flattened_x = x.permute(0, 2, 1, 3).flatten(1)
        # flattened_y = torch.linalg.solve(M, flattened_x)
        # y = flattened_y.reshape(B, H, C, W).permute(0, 2, 1, 3)
        
        # return y
        return self.reverse_level1(x)

    def reverse_level1(self, x):
        logdet_accum = 0

        out, logdet = self.conv_tl.reverse(x)
        logdet_accum += logdet
       
        return out

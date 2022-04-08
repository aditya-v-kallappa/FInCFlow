__author__ = 'aditya'

import torch
import torch.nn as nn

from layers.conv import PaddedConv2d, Conv1x1


class FastFlowUnit(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size):
        super(FastFlowUnit, self).__init__()
        if isinstance(kernel_size, int) or len(kernel_size) == 1:
            kernel_size = (kernel_size, kernel_size)
        assert in_channels % 4 == 0, "Input channels have to be a multiple of 4" 

        out_channels = in_channels // 4
        
        
        self.conv_tl = PaddedConv2d(out_channels, out_channels, kernel_size, order='TL')
        self.conv_tr = PaddedConv2d(out_channels, out_channels, kernel_size, order='TR')
        self.conv_bl = PaddedConv2d(out_channels, out_channels, kernel_size, order='BL')
        self.conv_br = PaddedConv2d(out_channels, out_channels, kernel_size, order='BR')
        

    
    def forward(self, x, context=None):
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)

        logdet_accum = 0.0

        out_tl, logdet = self.conv_tl.forward(x1)
        logdet_accum += logdet
        
        out_tr, logdet = self.conv_tr.forward(x2)
        logdet_accum += logdet
        
        out_bl, logdet = self.conv_bl.forward(x3)
        logdet_accum += logdet
        
        out_br, logdet = self.conv_br.forward(x4)
        logdet_accum += logdet

        out = torch.cat([out_tl, out_tr, out_bl, out_br], dim=1)
    
        return out, logdet_accum

    def reverse(self, x, context=None):

        logdet_accum = 0
        
        out_tl, out_tr, out_bl, out_br = torch.chunk(x, 4, dim=1)

        out_tl, logdet = self.conv_tl.reverse(out_tl)
        logdet_accum += logdet
       
        out_tr, logdet = self.conv_tr.reverse(out_tr)
        logdet_accum += logdet
        
        out_bl, logdet = self.conv_bl.reverse(out_bl)
        logdet_accum += logdet
        
        out_br, logdet = self.conv_br.reverse(out_br)
        logdet_accum += logdet
        
        out = torch.cat([out_tl, out_tr, out_bl, out_br], dim=1)

        return out

__author__ = 'aditya'

import torch
import torch.nn as nn

from conv import PaddedConv2d, Conv1x1


class FastFlowUnit(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size):
        super(FastFlowUnit, self).__init__()
        if isinstance(kernel_size, int) or len(kernel_size) == 1:
            kernel_size = (kernel_size, kernel_size)
        assert in_channels % 4 == 0, "Input channels have to be a multiple of 4" 

        out_channels = in_channels // 4
        
        self.split_input = torch.chunk
        self.conv_tl = PaddedConv2d(out_channels, out_channels, kernel_size, order='TL')
        self.conv_tr = PaddedConv2d(out_channels, out_channels, kernel_size, order='TR')
        self.conv_bl = PaddedConv2d(out_channels, out_channels, kernel_size, order='BL')
        self.conv_br = PaddedConv2d(out_channels, out_channels, kernel_size, order='BR')
        self.cat = torch.cat
        self.conv1x1 = Conv1x1(in_channels, in_channels, kernel_size=1, bias=True)

    
    def forward(self, x):
        x1, x2, x3, x4 = self.split_input(x, 4, dim=1)

        logdet_accum = 0.0

        out_tl, logdet = self.conv_tl.forward(x1)
        logdet_accum += logdet
        
        out_tr, logdet = self.conv_tr.forward(x2)
        logdet_accum += logdet
        
        out_bl, logdet = self.conv_bl.forward(x3)
        logdet_accum += logdet
        
        out_br, logdet = self.conv_br.forward(x4)
        logdet_accum += logdet

        out = self.cat([out_tl, out_tr, out_bl, out_br], dim=1)
        print(out.shape)
        out, logdet = self.conv1x1(out)
        logdet_accum += logdet

        return out, logdet_accum

    def inverse(self, x):

        logdet_accum = 0

        x, logdet = self.conv1x1.inverse(x)
        logdet_accum += logdet

        out_tl, out_tr, out_bl, out_br = self.split_input(x, 4, dim=1)

        out_tl, logdet = self.conv_tl.inverse(out_tl)
        logdet_accum += logdet
       
        out_tr, logdet = self.conv_tr.inverse(out_tr)
        logdet_accum += logdet
        
        out_bl, logdet = self.conv_bl.inverse(out_bl)
        logdet_accum += logdet
        
        out_br, logdet = self.conv_br.inverse(out_br)
        logdet_accum += logdet
        
        out = self.cat([out_tl, out_tr, out_bl, out_br], dim=1)

        return out, logdet_accum

import torch.nn as nn
from torch.nn.functional import pad, conv

class our_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = 1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.conv_tl 
        self.conv_br
        self.conv_tr
        self.conv_br
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        a,b,c,d = x.split(4)
        ap = self.conv1(pad(a))
        bp = self.conv2(pad(b))
        cp = self.conv1(pad(c))
        dp = self.conv2(pad(d))
        y = torch.cat([ap,bp,cp,dp])
        return self.pointwise(y)
        
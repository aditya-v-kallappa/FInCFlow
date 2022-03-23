import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


def _log_prob(loc, scale, value):
    # # if self._validate_args:
    # #     self._validate_sample(value)
    # # compute the variance
    # # var = (scale ** 2)
    # var = scale * scale
    # log_scale = math.log(scale) #if isinstance(scale, ) else scale.log()
    # return -((value - loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

    """
    value : B X d
    """
    loss = []
    for v in value:
        loss.append(const_k + 0.5 * (torch.linalg.norm(v) ** 2))
    loss = torch.tensor(loss, requires_grad=True).reshape(value.shape[0], -1).to(device)
    return loss



def _construct_matrix(x, conv_w):
    B, C, H, W = x.shape
    C_out, C_in, k_h, k_w = conv_w.shape
    M = torch.zeros(size=(C * H * W, C * H * W), dtype=x.dtype)
    
    flattened_w = torch.flatten(conv_w).flip(0)
    submatrix = torch.zeros(size=(H, W, W))

    # for i in range(C * H * W):
    #     for j in range(C * H * W):
    #         M[i, j] = 
    # for c in range(C):
    for i in range(k_h):#(H):
        for j in range(W):
            for k in range(j+1):
                if j - k >= k_w: #or i >= k_h:
                    continue
                submatrix[i, j, k] = flattened_w[k_w * i + j - k]

    for i in range(H):
        for j in range(i + 1):
            # start_w = i * (W + 1) 
            M[W*i:W*(i+1), W*j:W*(j+1)] = submatrix[i - j]
    

    return M

class InvertibleConv2d(nn.Module):
    def __init__(self, n_input_channels, bias=False):
        super().__init__()
        self.pad = nn.ConstantPad2d((2, 0, 2, 0), 0)
        self.conv = nn.Conv2d(n_input_channels, 1, (3, 3), 1, padding=0, bias=bias)
        self.init_weight()
        self.M = None

    def init_weight(self):
        self.conv.weight.data[:, -1, -1, -1] = torch.tensor([1.0]).to(dtype=self.conv.weight.dtype)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        logdet = 0
        return x, logdet

    def reverse(self, x, construct_matrix=False):
        B, C, H, W = x.shape 
        bias = 0.0 if not self.conv.bias else self.conv.bias.data

        if construct_matrix or not self.M:
            # W = torch.zeros(size=(H*W*C, H*W*C), dtype=x.dtype)
            self.M = _construct_matrix(x, self.conv.weight.data)
        y = []
        inv_M = torch.inverse(self.M)
        tensor_y = torch.zeros_like(x)
        for i, x_batch in enumerate(x):
            # y.append(
            #     torch.linalg.solve_triangular(M, (x_batch.flatten() - bias), upper=False,unitriangular = True).reshape(x_batch.shape))
            # y.append((inv_M @ (torch.flatten(x_batch, 0, -1) - bias)).reshape(x_batch.shape))
            tensor_y[i] = (inv_M @ (torch.flatten(x_batch, 0, -1) - bias)).reshape(x_batch.shape)
        # y = torch.tensor(*y).reshape(x.shape)
        logdet = 0
        return tensor_y, logdet

    def clear_grad(self):
        self.conv.weight.grad[:, -1, -1, -1] = 0.0


class Model2(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.n_layers = n_layers
        layers = []

        for _ in range(self.n_layers):
            layers.append(InvertibleConv2d(1, False))
        
        # self.layers = nn.ModuleList([InvertibleConv2d(1, False), InvertibleConv2d(1, False)])
        self.layers = nn.ModuleList(layers)
        self.flatten = torch.flatten
    
    def forward(self, x):
        logdet = 0
        xs = []
        for layer in self.layers:
            x, _logdet = layer(x)
            xs.append(x)
            logdet += _logdet
        
        x = self.flatten(x, 1, -1)

        return x, logdet
    
    def reverse(self, x, construct_matrix=True):
        logdet = 0
        xs = []
        for layer in self.layers:
            x, _logdet = layer.reverse(x, construct_matrix)
            xs.append(x)
            logdet += _logdet
        
        return x, logdet
    
    def clear_grad(self):
        for layer in self.layers:
            layer.clear_grad()
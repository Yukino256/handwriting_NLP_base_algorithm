from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# print(torch.ones(1).shape)
# print(torch.ones(1))
# [batch_size, seq_len, hidden_size]  or [batch_size, channels, height, width]
class LayerNorm(nn.Module):
    def __init__(self, *, 
                eps: float = 1e-6, 
                elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.gain = nn.Parameter(torch.ones(1))
            self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor):
        mean = x.mean(dim = 1, keepdim = True)
        mean_x2 = (x ** 2).mean(dim = 1, keepdim = True)
        variance = mean_x2 - mean ** 2
        x_norm = (x - mean) / torch.sqrt(variance + self.eps)
        if self.elementwise_affine:
            x_norm = self.gain * x_norm + self.bias

        return x_norm

test = torch.randn(2, 512, 1024)
norm = LayerNorm()
test_norm = norm(test)
print(test_norm.shape)
print(test_norm)
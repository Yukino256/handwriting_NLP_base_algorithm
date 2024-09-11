import torch
import torch.nn as nn
import torch.nn.functional as F
'''
公式是 x/ rms(x)
rms(x) = sqrt(mean(x^2) + eps)
'''
class RMSNorm(nn.Module):
    def __init__(self, eps, dim=1, elementwise_affine=True):
        super().__init__()
        self.eps = eps  
        self.dim = dim
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine: # 是否加入可学习参数
            self.gamma = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor):
        rmsx = x ** 2 # 
        rmxs = x.mean(dim=self.dim, keepdim=True)
        rms_norm = x * torch.rsqrt(rmsx + self.eps)
        if self.elementwise_affine:
            rms_norm = self.gamma * rms_norm
        return rms_norm

norm = RMSNorm(0.001)
test = torch.randn(2, 512, 1024)
print(test.shape)
print(test[0, 5, :10])
norm_test = norm(test)
print(norm_test.shape)
print(norm_test[0, 5, :10]) 

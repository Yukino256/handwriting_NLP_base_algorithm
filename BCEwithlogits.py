import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
def sigmoid(x: Tensor):
    return 1 / (1 + torch.exp(-x))

test = test_input = torch.randn(2, 512, 512)
test = sigmoid(test)
print(test)
print(test.shape)

class BCEwithlogits(nn.Module):
    def __init__(self, weights=1, reduction='mean'):
        super().__init__()
        self.weights = weights
        self.reduction = reduction

    def forward(self, input_tensor, target):
        input_tensor = torch.sigmoid(input_tensor)
        print(input_tensor.shape) # torch.Size([2, 512, 512])
        loss = - self.weights * torch.log(input_tensor) * target
        if self.reduction == 'mean':
            loss = loss.mean(-1) # 不指定dim就会只剩一个值
        elif self.reduction =='sum':
            loss = loss.sum(-1)
        return loss

test_input = torch.randn(2, 512, 512)
test_output = torch.eye(512).unsqueeze(0).repeat(2, 1, 1)

bce_loss = BCEwithlogits()

loss = bce_loss(test_input, test_output)
print(loss)
print(loss.shape) # torch.Size([2, 512]) batche_size x seq_length



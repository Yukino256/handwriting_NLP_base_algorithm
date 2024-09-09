import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def softmax(inputs: Tensor) -> Tensor:
    """
    Compute the softmax of the input tensor.
    """
    # in_shape = inputs.shape # 记录输入的shape
    # exp_x = torch.exp(inputs - torch.max(inputs, dim=1)[0].unsqueeze(1).expand(in_shape)) #输入的shape逐行减去这个max

    exp_x = torch.exp(inputs - torch.max(inputs, dim=1, keepdim=True)[0])
    return exp_x / torch.sum(exp_x, dim = 1, keepdim=True)
# 测试，计算结果是对的
# test = torch.tensor([[1,2,3],[4,5,6]])
# test2 = torch.randn(2, 512, 512)
# s1 = softmax(test)
# s2 = softmax(test2)
# print(s1)
# print(s1.shape)
# print(s2)
# print(s2.shape)

class CrossEntropyLoss(nn.Module):
    def __init__(self, weights=1, reduction='mean', epsilon=1e-6):
        super().__init__()
        
        self.weights = weights
        self.reduction = reduction
        self.epsilon = epsilon
        
    def forward(self, inputs, target):
        '''
        inputs: (batch_size, seq_len, class_nums)
        target: (batch_size, seq_len, class_nums)
        '''
        print(inputs.shape)
        inputs = softmax(inputs)
        print(inputs.shape)
        loss = -torch.log(inputs) + self.weights * target
        if self.reduction == 'mean':
            loss = torch.mean(loss)
            
        elif self.reduction =='sum':
            loss = torch.sum(loss)
            
        return loss

test = torch.randn(2, 512, 1024)
target = torch.randint(0, 2, (2, 512, 1024))
print(target.shape)
loss_func = CrossEntropyLoss()
loss = loss_func(test, target)
print(loss)
print(loss.shape)

        
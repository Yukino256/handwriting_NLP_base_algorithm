# 二分类交叉熵，需要计算真实值和预测值之间的交叉熵，公式如下：
# loss = -y * log(p) - (1-y) * log(1-p)
# 其中y是真实值，p是预测值，log是自然对数。

import torch.nn as nn
import torch
import torch.nn.functional as F

class BinaryCrossEntropyLoss(nn.Module): # 这个交叉熵面向的对象是两个一维的list，分别是预测值和真实值
    def __init__(self, predictions_with_sigmoid, labels):
        self.predictions = predictions_with_sigmoid
        self.labels = labels

    def forward():
        loss = 0
        for i in range(len(self.predictions)):
            if self.labels[i] == 1:
                loss -= torch.log(self.predictions[i])
        return loss

class BinaryCrossEntropyLoss(nn.Module): 
    '''
    这个交叉熵面向的是一个size为  [batch_size, q_len, num_heads * head_dim]  的tensor，或者其他shape的tensor
    '''
    def __init__(self, pos_weight=1, reduction='mean'):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, input_tensor, labels):
        '''
        input_tensor: [batch_size, q_len, num_heads * head_dim]
        是[batch_size, q_len, hidden_size]的输入，经过数个block之后，再加一个MLP之后的结果，shape保持不变
        input_tensor还需要经过一个sigmod函数，变成[batch_size, q_len, output_size]的tensor
        而labels: [batch_size, q_len, output_size]， 前者是一个logits，后者是一个one-hot的形式
        '''
        loss = -self.pos_weight * input_tensor * labels * torch.log(input_tensor)
        if self.reduction == 'mean':
            loss = loss.mean(-1)
        elif self.reduction == 'sum':
            loss = loss.sum(-1)
        return loss

# from torch.nn import functional as F
test_input = torch.randn(2, 512, 512)
# print(test_input[0, 0])
test_input = torch.sigmoid(test_input)
# print(test_input[0, 0])

# test_labels = torch.randint(0, 2, size=(2, 512, 512)) # 以生成int range(0, 2)
num_classes = 512
test_labels = torch.arange(num_classes)
test_labels = F.one_hot(test_labels, num_classes=num_classes)
test_labels = test_labels.unsqueeze(0).repeat(2, 1, 1)
# test_labels = test_labels.expand(2, 512, 512) 与上面等效
'''
等效写法：
test = torch.eye(512).unsqueeze(0).repeat(2, 1, 1)
# ！！！注意！expand只是数据的复用，repeat是真正占用了额外的空间！
# unsequeeze(0)是增加一个维度，就是多加一个[]，后续需要结合repeat或者expand使用
'''

print(test_labels.shape) [2, 512, 512] # 每个token维度都是one-hot形式编码
print(test_labels[0, 0]) # [1, 512]

loss_func = BinaryCrossEntropyLoss()
loss = loss_func(test_input, test_labels)

print(loss)
'''
tensor([[0.3012, 0.2973, 0.2966,  ..., 0.3014, 0.3046, 0.3003],
        [0.2968, 0.2955, 0.3022,  ..., 0.2945, 0.2950, 0.2946]])
'''
print(loss.shape) # torch.Size([2, 512])
loss = torch.mean(loss, dim = -1) 

# loss.backward() # element 0 of tensors does not require grad and does not have a grad_fn


print(loss) # tensor([0.2997, 0.2999])
print(loss.shape) # torch.Size([2])

        


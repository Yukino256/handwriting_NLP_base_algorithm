import torch
import torch.nn as nn
import torch.functional as F

# test = torch.rand(10, 512, 1024)
# test = test.view(-1, 256, 256)
# print(test.shape)
test = torch.rand(10, 512, 1024)
# print(test.size())

class linear(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.randn(input_size, output_size))
        self.bias = nn.Parameter(torch.randn(output_size))
        self.activations = nn.ReLU()

    def forward(self, x):
        # x [batch_size, seq_len, input_size]
        # weight [input_size, output_size]
        bz, seq_len, _ = x.shape
        x = x.view(-1, self.input_size)
        # print(x.shape)
        x = torch.mm(x, self.weight) + self.bias
        # print(x.shape)
        x = self.activations(x)
        x = x.reshape(bz, seq_len, -1)
        # print(x.shape)
        return x
    
# test = torch.rand(10, 512, 1024)
# linear_layer = linear(1024, 2048)
# test = linear_layer(test)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layers = nn.ModuleList()
        self.layers.append(linear(input_size, hidden_size))
        for i in range(self.hidden_layers):
            self.layers.append(linear(hidden_size, hidden_size))
        self.layers.append(linear(hidden_size, output_size))

    def forward(self, x):
        bcz, seq_len, _ = x.shape  # 记录shape  这里应该是记录前两个shape，后面的shape都可以融合到一起，在外面模型设计的时候再通过view/reshape分开
        print(self.layers)
        x = x.view(-1, 1,self.input_size)
        print(x.shape)
        for layer in self.layers:  # 也可以前面不定义模型，而是在这里
            x = layer(x)
        print(x.shape)
        x = x.view(bcz, seq_len, -1)  
        print(x.shape)
        return x

test = torch.rand(10, 512, 1024)
mlp_layer = MLP(1024, 20, 2048, 1024)
test = mlp_layer(test)

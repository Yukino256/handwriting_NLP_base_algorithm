import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.encodings = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)


        two_i = torch.arange(0, d_model, 2, dtype=torch.float)
        div_term = torch.pow(10000, two_i / d_model)
        # print(div_term.shape)
        div_term = two_i / div_term
        self.encodings[:, 0::2] = torch.sin(position * div_term)
        self.encodings[:, 1::2] = torch.cos(position * div_term)


    def forward(self, x: torch.Tensor):
        bcz, seq_len, _ = x.shape
        pe = self.encodings[:seq_len, :].unsqueeze(0).repeat(bcz, 1, 1)
        print(pe.shape)
        x = x + pe
        return x
    
max_len = 5000
d_model = 512
test = torch.randn(3, 1582, 512)
pe = PositionalEncoding(d_model, max_len)
test = pe(test)
print(test.shape)

class Transformer(nn.Module):
    pass
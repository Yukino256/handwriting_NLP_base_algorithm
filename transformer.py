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
    
# max_len = 5000
# d_model = 512
# test = torch.randn(3, 1582, 512)
# pe = PositionalEncoding(d_model, max_len)
# test = pe(test)
# print(test.shape)


class TransformerAttention(nn.Module):
    def __init__(self, hidden_size, head_nums, head_dims):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_nums = head_nums
        self.head_dims = head_dims
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        # x: [bcz, seq_len, hidden_size]
        bcz, seq_len, _ = x.shape
        # pe = PositionalEncoding(768)  # 这里加入绝对位置编码,实际上应该在embedding之前加
        # x = pe(x)
        q = self.q_proj(x).view(bcz, seq_len, self.head_nums, self.head_dims)
        if y is None:
            k = self.k_proj(x).view(bcz, seq_len, self.head_nums, self.head_dims)
            v = self.v_proj(x).view(bcz, seq_len, self.head_nums, self.head_dims)
        else:
            k = self.k_proj(y).view(bcz, seq_len, self.head_nums, self.head_dims)
            v = self.v_proj(y).view(bcz, seq_len, self.head_nums, self.head_dims)
        k = k.transpose(-2, -1)
        attn_weigths = torch.matmul(q, k) / math.sqrt(self.hidden_size)

        attn_output = torch.matmul(attn_weigths, v)

        attn_output = torch.softmax(attn_output, dim = 1).view(bcz, seq_len, -1)
        
        return attn_output

        
# test = torch.randn(5, 512, 768)
# attn = TransformerAttention(768, 12, 64)
# test = attn(test)
# print(test[:, :, :128])
# print(test.shape)

class Transformer(nn.Module):
    def __init__(self, 
                 encoder_layers: int = 12, 
                 decoder_layers:int = 12, 
                 hidden_size:int = 768, 
                 num_heads:int = 12, 
                 heads_dim:int = 64, 
                 drop_out :float = 0.1, 
                 mask: torch.Tensor =None):
        super().__init__()
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.heads_dim = heads_dim

        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.drop_out = nn.Dropout(drop_out)
        self.mask = mask
        # self.norm = nn

        self.encoders = nn.ModuleList(
                [
                    nn.Sequential(
                        TransformerAttention(hidden_size, num_heads, heads_dim), 
                        nn.Linear(hidden_size, hidden_size)
                    ) for _ in range(encoder_layers)
                ]
            )
        self.decoders = nn.ModuleList(
                [
                    nn.Sequential(
                        TransformerAttention(hidden_size, num_heads, heads_dim), 
                        nn.Linear(hidden_size, hidden_size), 
                        TransformerAttention(hidden_size, num_heads, heads_dim), 
                        nn.Linear(hidden_size, hidden_size)
                    ) for _ in range(decoder_layers)
                ]
            )

    def forward(self, x: torch.Tensor):
        bcz, s_len, _ = x.shape
        pe = PositionalEncoding(768)
        norm = nn.LayerNorm([s_len, self.hidden_size])

        x = pe(x)
        for attn, ffn in self.encoders:
            residual = x
            x = self.drop_out(attn(x))
            x += residual

            x = norm(x)
            residual = x
            x = self.drop_out(ffn(x))
            x += residual
            x = norm(x)

        encoder_res = x

        for attn, ffn, attn_2, ffn_2 in self.decoders:

            if self.mask is not None:
                x += self.mask
            residual = x
            x = self.drop_out(attn(x))
            x += residual
            x = norm(x)  # 第一个attn

            residual = x
            x  = self.drop_out(attn_2(x, encoder_res))
            x += residual
            x = norm(x)  # 用了encoder的attn

            residual = x
            x = self.drop_out(ffn(x))
            x += residual
            x = norm(x)
        
        return x

test = torch.randn(5, 512, 768)
attn = Transformer(12, 12, 768, 12, 64)
test = attn(test)
print(test[:, :, :128])
print(test.shape)



        


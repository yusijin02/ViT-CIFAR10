import torch.nn as nn
import torch
import math
import sys
sys.path.append("..")
from einops import rearrange
from Models.DropPath import DropPath
from flags import parser

class MultiHeadAttention(nn.Module):
    # Input :  torch.Size([32, 197, 768])
    #          32  = batch_size
    #          197 = 196 + 1 = 14 * 14 + 1 = nums_patch + cls_token = (224/16) * (224/16) + 1
    #          768 = hidden_size 
    # Output:  torch.Size([32, 197, 768])
    def __init__(self, FLAGS):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = FLAGS.hidden_size
        self.layernorm = nn.LayerNorm(FLAGS.hidden_size, FLAGS.eps)
        self.d_q = self.d_k = self.d_v = FLAGS.embedding_dim
        self.num_heads = FLAGS.num_heads
        self.W_q = nn.Linear(FLAGS.hidden_size, self.d_q * self.num_heads)
        self.W_k = nn.Linear(FLAGS.hidden_size, self.d_k * self.num_heads)
        self.W_v = nn.Linear(FLAGS.hidden_size, self.d_v * self.num_heads)
        self.softmax = nn.Softmax(dim=-1)
        self.drop_path = DropPath(FLAGS.drop_prob)
    
    def forward(self, x):
        res = x  # torch.Size([64, 17, 256])
        # print(x.shape)
        query = self.W_q(x).view(x.shape[0], -1, self.num_heads, self.d_q).transpose(1, 2)  # torch.Size([64, 12, 17, 64])
        key = self.W_k(x).view(x.shape[0], -1, self.num_heads, self.d_k).transpose(1, 2)    # torch.Size([64, 12, 17, 64])
        value = self.W_v(x).view(x.shape[0], -1, self.num_heads, self.d_v).transpose(1, 2)  # torch.Size([64, 12, 17, 64])
        # print(key.shape)
        attention = torch.matmul(query, key.transpose(-1, -2))  # torch.Size([64, 12, 17, 17])
        attention = attention / math.sqrt(self.d_k)  # torch.Size([64, 12, 17, 17])
        attention = self.softmax(attention)  # torch.Size([64, 12, 17, 17])
        context = torch.matmul(attention, value)  # torch.Size([64, 12, 17, 64])
        context = context.transpose(1, 2)  # torch.Size([64, 17, 12, 64])
        context = context.reshape(x.shape[0], -1, self.hidden_size)  # torch.Size([64, 51, 256])
        context = context + self.drop_path(context)  # torch.Size([64, 51, 256])
        x = self.layernorm(res + context)  
        return x
    


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    x = torch.rand(32, 197, 768)
    mha = MultiHeadAttention(FLAGS)
    mha(x)
    
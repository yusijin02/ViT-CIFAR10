import torch.nn as nn
from Models.MLP import MLP
from Models.MultiHeadAttention import MultiHeadAttention

class TransformerBlock(nn.Module):
    # Input :  torch.Size([32, 197, 768])
    #          32  = batch_size
    #          197 = 196 + 1 = 14 * 14 + 1 = nums_patch + cls_token = (224/16) * (224/16) + 1
    #          768 = hidden_size 
    # Output:  torch.Size([32, 197, 768])
    def __init__(self, FLAGS):
        super(TransformerBlock, self).__init__()
        self.mlp = MLP(FLAGS)
        self.attention = MultiHeadAttention(FLAGS)
        
    def forward(self, x):
        x = self.attention(x)
        x = self.mlp(x)
        return x
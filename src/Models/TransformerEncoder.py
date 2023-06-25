import torch.nn as nn
import copy
from TransformerBlock import TransformerBlock

class TransformerEncoder(nn.Module):
    # Input :  torch.Size([32, 197, 768])
    #          32  = batch_size
    #          197 = 196 + 1 = 14 * 14 + 1 = nums_patch + cls_token = (224/16) * (224/16) + 1
    #          768 = hidden_size 
    # Output:  torch.Size([32, 197, 768])
    def __init__(self, FLAGS):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(FLAGS.num_layers):
            layer = TransformerBlock(FLAGS)
            self.layers.append(copy.deepcopy(layer))
        self.layernorm = nn.LayerNorm(FLAGS.hidden_size, FLAGS.eps)
    
    def forward(self, x):
        for layer_block in self.layers:
            x = layer_block(x)
        x = self.layernorm(x)
        return x
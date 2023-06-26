import torch.nn as nn

from Models.DropPath import DropPath

class MLP(nn.Module):
    # Input :  torch.Size([32, 197, 768])
    #          32  = batch_size
    #          197 = 196 + 1 = 14 * 14 + 1 = nums_patch + cls_token = (224/16) * (224/16) + 1
    #          768 = hidden_size 
    # Output:  torch.Size([32, 197, 768])
    def __init__(self, FLAGS):
        super(MLP, self).__init__()
        self.layernorm = nn.LayerNorm(FLAGS.hidden_size, FLAGS.eps)
        self.fc_1 = nn.Linear(FLAGS.hidden_size, FLAGS.mlp_hidden_size)
        self.fc_2 = nn.Linear(FLAGS.mlp_hidden_size, FLAGS.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(FLAGS.dropout_rate)
        self.drop_path = DropPath(FLAGS.drop_prob)
    
    def forward(self, x):
        res = x
        x = self.layernorm(x)
        x = self.fc_1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        # x = self.dropout(x)
        x = x + self.drop_path(x)
        x = self.layernorm(res + x)
        return x
import torch.nn as nn
from einops import rearrange
from DropPath import DropPath

class MultiHeadAttention(nn.Module):
    def __init__(self, FLAGS):
        super(MultiHeadAttention, self).__init__()
        self.layernorm = nn.LayerNorm(FLAGS.hidden_size, FLAGS.eps)
        d_q, d_k, d_v = FLAGS.embedding_dim
        num_heads = FLAGS.num_heads
        self.W_q = nn.Linear(FLAGS.hidden_size, d_q * num_heads)
        self.W_k = nn.Linear(FLAGS.hidden_size, d_k * num_heads)
        self.W_v = nn.Linear(FLAGS.hidden_size, d_v * num_heads)
        self.softmax = nn.Softmax(dim=-1)
        self.drop_path = DropPath(FLAGS.drop_prob)
    
    
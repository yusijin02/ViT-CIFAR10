import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, FLAGS):
        super(MLP, self).__init__()
        self.layernorm = nn.LayerNorm(FLAGS.hidden_size, FLAGS.eps)
        self.fc_1 = nn.Linear(FLAGS.hidden_size, self.mlp_hidden_size)
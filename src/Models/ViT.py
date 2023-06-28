import torch
import torch.nn as nn
import os
from Models.PatchEmbedding import PatchEmbedding
from Models.TransformerEncoder import TransformerEncoder

import torch
import torch.nn as nn
from torch.nn import Transformer

class VisionTransformer(nn.Module):
    def __init__(self, FLAGS):
        super(VisionTransformer, self).__init__()
        self.flags = FLAGS
        self.patch_embed = nn.Conv2d(in_channels=self.flags.img_chans, out_channels=self.flags.hidden_size, 
                                     kernel_size=self.flags.patch_size, stride=self.flags.patch_size, bias=False)
        num_patch = (self.flags.img_size // self.flags.patch_size) ** 2
        num_cls_token = 1
        self.cls_token = nn.Parameter(torch.zeros(1, FLAGS.hidden_size, 1)) 
        self.num_tokens = num_cls_token + num_patch
        self.position_enc = nn.Parameter(torch.zeros(1, self.num_tokens, self.flags.hidden_size))
        self.dropout = nn.Dropout(self.flags.dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.flags.hidden_size, nhead=self.flags.num_heads,
                                                   dim_feedforward=self.flags.feedforward_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.flags.num_layers)

        self.fc = nn.Linear(self.flags.hidden_size, self.flags.num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
                                      # input images   =====>   # [batch_size, in_chans, H, W]
        x = self.patch_embed(x)                                 # [batch_size, hidden_size, num_patches**0.5, num_patches**0.5]
        x = x.flatten(2)                                        # [batch_size, hidden_size, num_patches]
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # [batch_size, hidden_size, 1]
        x = torch.cat((cls_tokens, x), dim=2)                   # [batch_size, hidden_size, num_patches+1]
        x = x.permute(0, 2, 1)                                  # [batch_size, num_patches+1, hidden_size]
                                 # self.position_enc   =====>   # [1, num_patches+1, hidden_size]
        x = x + self.position_enc                               # [batch_size, num_patches+1, hidden_size]
        x = self.dropout(x)                                     # [batch_size, num_patches+1, hidden_size]
        x = self.transformer_encoder(x)                         # [batch_size, num_patches+1, hidden_size]
        x = x[:, 0, :]                                          # [batch_size, hidden_size]
        x = self.fc(x)              # output classes   =====>   # [batch_size, num_classes]
        x = self.softmax(x)                                     # 灵异事件滚出中国
        return x                                                # 🐱🐱保佑平安, acc上98

        
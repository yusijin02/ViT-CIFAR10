import torch
import torch.nn as nn
import os
from Models.PatchEmbedding import PatchEmbedding
from Models.TransformerEncoder import TransformerEncoder

class ViT(nn.Module):
    def __init__(self, FLAGS):
        super(ViT, self).__init__()
        self.patch_embedding = PatchEmbedding(FLAGS)  # [32, 3, 224, 224] ===> [32, 196, 768]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, FLAGS.hidden_size))  # [1, 1, 768]
        num_patches = (int(FLAGS.picture_size / FLAGS.patch_size)) ** 2  # 196
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, FLAGS.hidden_size))  # [1, 197, 768]
        self.encoder = TransformerEncoder(FLAGS)  # [32, 197, 768] ===> [32, 197, 768]
        self.fc1 = nn.Linear(FLAGS.hidden_size, FLAGS.cls_hidden)  # [32, 768] ===> [32, 10]
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(FLAGS.cls_hidden, FLAGS.num_classes)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # print(x.shape)
        x = self.patch_embedding(x)  # [32, 3, 224, 224] ===> [32, 196, 768]
        # print(x.shape)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # [32, 1, 768]
        # print(cls_tokens.shape)
        x = torch.cat((cls_tokens, x), dim=1)  # [32, 197, 768]
        
        # print(x.shape)
        # print(self.pos_embed.shape)
        x = x + self.pos_embed  # [32, 197, 768] + [1, 197, 768] = [32, 197, 768]
        
        # pos_embed = self.pos_embed.unsqueeze(0).repeat_interleave(x.shape[0], dim=0)
        # x = torch.add(x, pos_embed)  # [32, 197, 768] + [1, 197, 768] = [32, 197, 768]
        # pos_embed.requires_grad = False
        
        x = self.encoder(x)  # [32, 197, 768] ===> [32, 197, 768]
        x = x[:, 0]  # [32, 197, 768] ===> [32, 768], 取每个197里的第0个
        x = self.fc(x)  # [32, 768] ===> [32, 10]
        x = self.softmax(x)  # [32, 10]
        return x
        
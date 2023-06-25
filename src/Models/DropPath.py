import torch
import torch.nn as nn

def drop_path(x, drop_prob: float = 0, training: bool = False):
    # 在训练的时候, 以 drop_prod 为概率随机丢弃一些样本, 丢弃的样本填充 0
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    # print(x.shape)  # torch.Size([32, 3, 224, 224])
    # print(shape)  # (32, 1, 1, 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)  # 均值为 keep_prob
    random_tensor.floor_()  # 向下取整用于确定保存哪些样本
    output = x.div(keep_prob) * random_tensor
    # print(output.shape)  # torch.Size([32, 3, 224, 224])
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob = 0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
        


        
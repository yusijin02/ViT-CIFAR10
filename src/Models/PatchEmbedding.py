import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    # Input :  torch.Size([32, 3, 224, 224])
    #          32  = batch_size
    #          3   = channels (RGB)
    #          224 = height (H)
    #          224 = weight (W)
    # Output:  torch.Size([32, 196, 768])
    #          32  = batch_size
    #          196 = 14 * 14 = (224/16) * (224/16)
    #          16  = patch_size
    #          768 = hidden_size
    def __init__(self, FLAGS):
        super(PatchEmbedding, self).__init__()
        in_chans = FLAGS.in_chans
        patch_size = FLAGS.patch_size
        hidden_size = FLAGS.hidden_size
        self.conv = nn.Conv2d(in_channels=in_chans, out_channels=hidden_size,
                              kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # print(x.shape)  # torch.Size([32, 3, 224, 224])
        x = self.conv(x).flatten(2).transpose(1, 2)
        # print(x.shape)  # torch.Size([32, 196, 768])
        return x

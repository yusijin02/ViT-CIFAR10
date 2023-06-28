import argparse
parser = argparse.ArgumentParser()

# input images
parser.add_argument("--img_size", type=int, default=32, help="the size of input image")
parser.add_argument("--num_classes", type=int, default=10, help="the number of classes of the images")
parser.add_argument("--img_chans", type=int, default=3, help="the number of channels of the images")
# model setting
parser.add_argument("--hidden_size", type=int, default=72, help="the hidden size of ViT")
parser.add_argument("--patch_size", type=int, default=8, help="the size of patch")
parser.add_argument("--num_heads", type=int, default=12, help="the number of heads in the multiheadattention")
parser.add_argument("--feedforward_dim", type=int, default=2048, help="the dimension of the feedforward network")
parser.add_argument("--num_layers", type=int, default=3, help="the number of transformer encoder layers")
# training setting
parser.add_argument("--dropout_rate", type=float, default=0.1, help="the dropout value")
parser.add_argument("--epoch", type=int, default=200, help="the number of epochs")
parser.add_argument("--batch_size", type=int, default=128, help="the size of small batch")
# enviroment setting
parser.add_argument("--cuda_visable_device", type=str, default="0", help="string of cuda visable device. These GPUs will be used")

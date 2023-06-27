import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--in_chans", type=int, default=3, help="numbers of input channels")
parser.add_argument("--patch_size", type=int, default=16, help="patch size")
parser.add_argument("--hidden_size", type=int, default=768, help="hidden size")
parser.add_argument("--embedding_dim", type=int, default=64, help="embedding dim, d_k, d_q, d_v")
parser.add_argument("--num_heads", type=int, default=12, help="number of heads")
parser.add_argument("--drop_prob", type=float, default=0.1, help="drop path rate")
parser.add_argument("--dropout_rate", type=float, default=0.1, help="dropout rate")

parser.add_argument("--num_layers", type=int, default=3, help="number of layers")
parser.add_argument("--picture_size", type=int, default=32, help="picture size")
parser.add_argument("--num_classes", type=int, default=10, help="number of classes")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--mlp_hidden_size", type=int, default=3072, help="hidden size of MLPs")

parser.add_argument("--eps", type=float, default=1e-6, help="epsilon, the infinitesimal numbers")
parser.add_argument("--adam_beta_1", "-b1", type=float, default=0.9, help="beta_1 of Adam")
parser.add_argument("--adam_beta_2", "-b2", type=float, default=0.999, help="beta_2 of Adam")
parser.add_argument("--steplr_step_size", type=int, default=20, help="step size of torch.optim.lr_scheduler.StepLR")
parser.add_argument("--steplr_gamma", type=float, default=0.95, help="gamma of torch.optim.lr_scheduler.StepLR")
parser.add_argument("--epoch", type=int, default=50, help="epoch")
parser.add_argument("--max_norm", type=float, default=0.5, help="max norm of weights")

parser.add_argument("--check_epoch", type=int, default=20, help="save checkpoints each _ epochs")
parser.add_argument("--num_gpus", type=int, default=1, help="number of using GPUs")
parser.add_argument("--cuda_visable_device", type=str, default='0', help="cuda visable device")

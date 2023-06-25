import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--in_chans", type=int, default=3, help="numbers of input channels")
parser.add_argument("--patch_size", type=int, default=16, help="patch size")
parser.add_argument("--hidden_size", type=int, default=768, help="hidden size")
parser.add_argument("--embedding_dim", type=int, default=64, help="embedding dim, d_k, d_q, d_v")
parser.add_argument("--nums_heads", type=int, default=12, help="nums of heads")
parser.add_argument("--drop_prob", type=float, default=0.2, help="drop path rate")

parser.add_argument("--eps", type=float, default=1e-6, help="epsilon, the infinitesimal numbers")


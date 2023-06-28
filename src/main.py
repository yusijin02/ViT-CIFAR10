from flags import parser

FLAGS = parser.parse_args()

from Models.ViT import VisionTransformer
import torch

model = VisionTransformer(FLAGS)
x = torch.rand(64, 3, 32, 32)
model(x)
from flags import parser
from Models.ViT import VisionTransformer
import os
import torch
from train import Experiment

FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_visable_device
FLAGS.device_ids = FLAGS.cuda_visable_device.split(",")
for index in range(FLAGS.device_ids):
    FLAGS.device_ids[index] = int(FLAGS.device_ids[index])

exp = Experiment(FLAGS)
exp.train()
exp.vaild()
exp.draw()
exp.log()
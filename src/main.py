import os

from flags import parser
from train import Experiment


FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_visable_device



exp = Experiment(FLAGS)

exp.train()
exp.vaild()
exp.draw()
exp.log()

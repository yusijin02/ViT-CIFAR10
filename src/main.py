import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from flags import parser
from train import Experiment


FLAGS = parser.parse_args()





exp = Experiment(FLAGS)

exp.train()
exp.vaild()
exp.draw()
exp.log()

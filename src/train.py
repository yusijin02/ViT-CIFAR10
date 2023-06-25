import time
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from datetime import datetime
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from Models.ViT import ViT
from flags import parser
FLAGS = parser.parse_args()

trans_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomGrayscale(),       # 随机灰度化处理
    transforms.ToTensor(),              # 转为Tensor格式
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])
trans_vaild = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

class Experiment:
    def __init__(self, FLAGS) -> None:
        self.root = "./data"
        train_set = datasets.CIFAR10(root=self.root, train=True,  download=True, transform=trans_train)
        vaild_set = datasets.CIFAR10(root=self.root, train=False, download=True, transform=trans_vaild)
        self.train_loader = DataLoader(dataset=train_set, batch_size=FLAGS.batch_size, shuffle=True)
        self.vaild_loader = DataLoader(dataset=vaild_set, batch_size=FLAGS.batch_size, shuffle=False)
        
        model = ViT(FLAGS)
        model = nn.DataParallel(model, device_ids=range(FLAGS.num_gpus))
        self.model = model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=FLAGS.learning_rate,
                                          betas=(FLAGS.adam_beta_1, FLAGS.adam_beta_2), eps=FLAGS.eps)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                         step_size=FLAGS.steplr_step_size, gamma=FLAGS.steplr_gamma)
        self.epoch = FLAGS.epoch
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.flags = FLAGS
        self.x = []
        self.y = []
        
        self.best_loss  = 100000
        self.best_epoch = None
        
        self._mkdir_logs_path()
    
    def train(self):
        for epoch in range(self.epoch):
            train_loss = 0.0
            index = 1
            correct = 0
            num = 0
            start_time = time.time()
            _n = 1
            
            for x, y in self.train_loader:
                x = x.cuda()
                y = y.cuda()
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y) / len(x)
                train_loss += loss.item()
                
                # 梯度回传
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.flags.max_norm)  # 权重修剪
                self.optimizer.step()
                
                correct += torch.sum(torch.argmax(y_hat, dim=1) == y).item()
                num += len(x)
                if _n >= 10000:
                    _n = 1
                    print(f"    index: {index:.3}, loss: {loss.item():.6f}, accuracy: {(100 * correct / num):.3f}%")
                index += 1
                _n += len(x)
            
            end_time = time.time()
            self.scheduler.step()
            self.y.append(train_loss)
            self.x.append(epoch + 1)
            
            if train_loss < self.best_loss:
                self.best_loss  = train_loss
                self.best_epoch = epoch
                now = datetime.now()
                path_best = os.path.join(self.log_dir, "best.pth")
                path_check = os.path.join(self.log_dir, f"checkpoints-{now.hour}-{now.minute}-{now.second}")
                torch.save(self.model, path_best)
                torch.save(self.model, path_check)
            print(f"Using Time: {end_time - start_time}s, Epoch: {epoch + 1}, Loss = {train_loss}, Accuracy = {100 * correct / num}")
                


    def vaild(self):
        num = 0
        correct = 0
        vaild_loss = 0.0
        for x, y in self.vaild_loader:
            x = x.cuda()
            y = y.cuda()
            y_hat = self.model(x)
            
            loss = self.criterion(y_hat, y) / len(x)
            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).item()
            num += len(x)
            vaild_loss += loss.item()
        print(f"Test Loss: {vaild_loss:.3f}, Test Accuracy: {100 * correct / num:.5f}%")
        
        


    def _mkdir_logs_path(self):
        now = datetime.now()
        self.log_dir = os.path.join("./logs", f"{now.year}-{now.month}-{now.day}", f"weight-{now.hour}{now.minute}")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
    def draw(self):
        pic_path = os.path.join(self.log_dir, "loss.png")
        fig = plt.figure(figsize=(20, 8), dpi=80)
        plt.plot(self.x, self.y)
        plt.xticks(self.x)
        plt.savefig(pic_path)



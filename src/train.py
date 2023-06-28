import time
import torch
import torch.nn as nn
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from Models.ViT import VisionTransformer


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
        self.root = "../data"
        train_set = datasets.CIFAR10(root=self.root, train=True,  download=True, transform=trans_train)
        vaild_set = datasets.CIFAR10(root=self.root, train=False, download=True, transform=trans_vaild)
        self.train_loader = DataLoader(dataset=train_set, batch_size=FLAGS.batch_size, shuffle=True)
        self.vaild_loader = DataLoader(dataset=vaild_set, batch_size=FLAGS.batch_size, shuffle=False)
        
        model = VisionTransformer(FLAGS)
        model = nn.DataParallel(model, device_ids=FLAGS.device_ids)
        self.model = model.cuda()
        # self.optimizer = torch.optim.Adam(self.model.parameters(),
        #                                   betas=(FLAGS.adam_beta_1, FLAGS.adam_beta_2), eps=FLAGS.eps)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
        #                                                  step_size=FLAGS.steplr_step_size, gamma=FLAGS.steplr_gamma)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=FLAGS.steplr_step_size)
        self.epoch = FLAGS.epoch
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.flags = FLAGS
        self.x = []
        self.y = []
        self.acc_list = []
        
        self.best_loss  = 100000
        self.best_epoch = None
        
        self._mkdir_logs_path()
    
    def train(self):
        self.model.train()
        for epoch in range(self.epoch):
            train_loss = 0.0
            index = 1
            correct = 0
            num = 0
            start_time = time.time()
            _n = 1
            
            for x, y in self.train_loader:
                # self._image_show(x, y)
                x = x.cuda()
                y = y.cuda()
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
                train_loss += loss.item()
                
                # 梯度回传
                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.flags.max_norm)  # 权重修剪
                self.optimizer.step()
                
                correct += torch.sum(torch.argmax(y_hat, dim=1) == y).item()
                num += len(x)
                if _n >= 10000:
                    _n = 1
                    print(f"    loss: {loss.item():.6f}, accuracy: {(100 * correct / num):.3f}%")
                index += 1
                _n += len(x)
            
            end_time = time.time()
            self.scheduler.step()
            self.y.append(train_loss)
            self.x.append(epoch + 1)
            
            if train_loss < self.best_loss:
                self.best_loss  = train_loss
                self.best_epoch = epoch
                self.path_best = os.path.join(self.log_dir, "best.pth")
                self.best_acc = 100 * correct / num
                torch.save(self.model, self.path_best)
            acc = 100 * correct / num
            self.acc_list.append(acc)
            print(f"Using Time: {end_time - start_time}s, Epoch: {epoch + 1}, Loss = {train_loss}, Accuracy = {acc}%")
            
            if epoch % self.flags.check_epoch == 0:
                checkpoint_path = os.path.join(self.log_checkpoint_dir, f"epoch_{epoch}.pth")
                torch.save(self.model, checkpoint_path)
        
        self.trainloss_final = train_loss
        self.trainacc_final = acc
        self.epoch_final = epoch
                


    def vaild(self, type="Final"):
        self.model.eval()
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
        acc = 100 * correct / num
        print(f"{type} Test Loss: {vaild_loss:.3f}, Test Accuracy: {acc:.5f}%")
        if type == "Final":
            self.test_loss_final = vaild_loss
            self.test_acc_final  = acc
        else:
            self.test_loss_best = vaild_loss
            self.test_acc_best  = acc
        
    def _image_show(self, x, y):
        for i in range(len(x)):
            tensorImg = x[i]
            label = y[i]
            arrayImg = tensorImg.numpy()
            arrayShow = np.squeeze(arrayImg, 0)
            plt.imshow(arrayShow)


    def _mkdir_logs_path(self):
        now = datetime.now()
        self.log_dir = os.path.join("../logs", f"{now.year}-{now.month}-{now.day}", f"{now.hour}-{now.minute}")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_checkpoint_dir = os.path.join(self.log_dir, "checkpoint")
        if not os.path.exists(self.log_checkpoint_dir):
            os.makedirs(self.log_checkpoint_dir)
    
    def draw(self):
        pic_path = os.path.join(self.log_dir, "loss.png")
        fig = plt.figure(figsize=(20, 8), dpi=80)
        plt.plot(self.x, self.y)
        plt.savefig(pic_path)
        
        pic_path = os.path.join(self.log_dir, "acc.png")
        fig = plt.figure(figsize=(20, 8), dpi=80)
        plt.plot(self.x, self.acc_list)
        plt.savefig(pic_path)
    
    def log(self):
        log_path = os.path.join(self.log_dir, "flag.json")
        with open(log_path, 'a') as f:
            json.dump(vars(self.flags), f)
        
        self.model = torch.load(self.path_best).cuda()
        self.vaild("Best")
        result_path = os.path.join(self.log_dir, "result.json")
        result = {
            "final": {
                "epoch": self.epoch_final,
                "train": {
                    "loss": self.trainloss_final,
                    "acc" : self.trainacc_final
                },
                "test": {
                    "loss": self.test_loss_final,
                    "acc" : self.test_acc_final
                }
            },
            "best":{
                "epoch": self.best_epoch,
                "train": {
                    "loss": self.best_loss,
                    "acc" : self.best_acc
                },
                "test": {
                    "loss": self.test_loss_best,
                    "acc" : self.test_acc_best
                }
            }
        }
        with open(result_path, 'a') as f:
            json.dump(result, f)




# -*- coding: utf-8 -*-
"""hw3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LRld8HhcWN2HqYuXPmDOpsVY5yoBVJrn
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import torch.optim as optim
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
f = open('output.txt', "w+")

class TemplateNet1(nn.Module):
    def __init__(self):
        super(TemplateNet1, self).__init__()
        #input size is 3,32,32 
        self.conv1 = nn.Conv2d(3, 128, 3) ## (A)
        #self.conv2 = nn.Conv2d(128, 128, 3) ## (B)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(28800, 1000) ## (C)
        self.fc2 = nn.Linear(1000, 10)
        self.total_correct = 0

    def forward(self, x):
        #from 3,32,32 to 128,32,32
        x = self.pool(F.relu(self.conv1(x)))
        ## Uncomment the next statement and see what happens to the
        ## performance of your classifier with and without padding.
        ## Note that you will have to change the first arg in the
        ## call to Linear in line (C) above and in the line (E)
        ## shown below. After you have done this experiment, see
        ## if the statement shown below can be invoked twice with
        ## and without padding. How about three times?
        ## x = self.pool(F.relu(self.conv2(x))) ## (D)
        x = x.view(-1, 28800) ## (E)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_num_correct(self, preds, labels):
      return preds.argmax(dim=1).eq(labels).sum().item()

#Task 2
class TemplateNet2(nn.Module):
    def __init__(self):
        super(TemplateNet2, self).__init__()
        #input size is 3,32,32 
        self.conv1 = nn.Conv2d(3, 128, 3) ## (A)
        self.conv2 = nn.Conv2d(128, 128, 3) ## (B)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4608, 1000) ## (C)
        self.fc2 = nn.Linear(1000, 10)
        self.total_correct = 0

    def forward(self, x):
        #from 3,32,32 to 128,32,32
        x = self.pool(F.relu(self.conv1(x)))
        ## Uncomment the next statement and see what happens to the
        ## performance of your classifier with and without padding.
        ## Note that you will have to change the first arg in the
        ## call to Linear in line (C) above and in the line (E)
        ## shown below. After you have done this experiment, see
        ## if the statement shown below can be invoked twice with
        ## and without padding. How about three times?
        x = self.pool(F.relu(self.conv2(x))) ## (D)
        x = x.view(-1, 4608) ## (E)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_num_correct(self, preds, labels):
      return preds.argmax(dim=1).eq(labels).sum().item()

#Task 3
class TemplateNet3(nn.Module):
    def __init__(self):
        super(TemplateNet3, self).__init__()
        #input size is 3,32,32 
        self.conv1 = nn.Conv2d(3, 128, 3, padding=1) ## (A)
        self.conv2 = nn.Conv2d(128, 128, 3) ## (B)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6272, 1000) ## (C)
        self.fc2 = nn.Linear(1000, 10)
        self.total_correct = 0

    def forward(self, x):
        #from 3,32,32 to 128,32,32
        x = self.pool(F.relu(self.conv1(x)))
        ## Uncomment the next statement and see what happens to the
        ## performance of your classifier with and without padding.
        ## Note that you will have to change the first arg in the
        ## call to Linear in line (C) above and in the line (E)
        ## shown below. After you have done this experiment, see
        ## if the statement shown below can be invoked twice with
        ## and without padding. How about three times?
        x = self.pool(F.relu(self.conv2(x))) ## (D)
        x = x.view(-1, 6272) ## (E)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_num_correct(self, preds, labels):
      return preds.argmax(dim=1).eq(labels).sum().item()

def run_code_for_training(net):
    #c = 0
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    #cmt = torch.zeros(10,10, dtype=torch.int64)
    for epoch in range(1):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            #print(inputs)
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            #k = outputs.argmax(1)
            #for a in range(4):
              #if k[a] == labels[a]:
                #c = c+1
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #net.total_correct += net.get_num_correct(outputs, labels)
            #stacked = torch.stack((labels, outputs.argmax(dim=1)),dim=1)
            #for p in stacked:
              #tl, pl = p.tolist()
              #cmt[tl, pl] = cmt[tl, pl] + 1

            if i % 2000 == 1999:
              if i == 11999:
                print("\n[epoch:%d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / float(2000)))
                f.write("[epoch:%d, batch:%5d] loss: %.3f \n" % (epoch + 1, i + 1, running_loss / float(2000)))          
                          
              running_loss = 0.0

    #print('Training accuracy: ', (c*100)/50000)
    #print(cmt)

def run_code_for_testing(net):
    #c = 0
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    cmt = torch.zeros(10,10, dtype=torch.int64)
    for epoch in range(1):
        running_loss = 0.0
        for i, data in enumerate(testloader):
            inputs, labels = data
            #print(inputs)
            inputs = inputs.to(device)
            labels = labels.to(device)
            #optimizer.zero_grad()
            outputs = net(inputs)
            #k = outputs.argmax(1)
            #for a in range(4):
              #if k[a] == labels[a]:
                #c = c+1
            loss = criterion(outputs, labels)
            #loss.backward()
            #optimizer.step()
            running_loss += loss.item()
            net.total_correct += net.get_num_correct(outputs, labels)
            stacked = torch.stack((labels, outputs.argmax(dim=1)),dim=1)
            for p in stacked:
              tl, pl = p.tolist()
              cmt[tl, pl] = cmt[tl, pl] + 1
            if i % 2000 == 1999:
                #print("\n[epoch:%d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / float(2000)))
                running_loss = 0.0
    print(cmt)
    f.write(str(cmt))

net1 = TemplateNet1()
net2 = TemplateNet2()
net3 = TemplateNet3()
run_code_for_training(net1)
run_code_for_training(net2)
run_code_for_training(net3)
run_code_for_testing(net3)
f.close()



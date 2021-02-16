#Credits: Prof. Kak #We are asked to design an augmented logic to LOADnet2 to improve the accuracy of 50% noise dataset. With most of my energy exhausted in task 3, I decided to make my filter learnable.
#I had been playing only with static filters all the while, now with freedom to tweak the LOADnet, I tried implementing my HW4 Resnet-69 architecture that I proposed to get output.
#It didn't fetch me good results. So, I decided to pass another value as output from the CNN, which is the standard deviation. I'll make the network learn the suitable sigma value after assigning the ground truth as 1.
#This technique helped in getting good accuracy in lesser epochs. That being said, with more time (Maybe i should've not done lot of research for task 3 and managed time efficiently), I can
#fetch much better results with this technique. Making sigma learnable is a masterstroke. Thanks to a forum discussion in pytorch, and my reading on Kornia filters. I was able to know that
#these two can be used. I felt the one in the pytorch forums was easier to comprehend and visualize. That infact did wonders. 

import random
import numpy
import torch
import os, sys
import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision                  
from torchvision import datasets
import torchvision.transforms as tvt
import torch.optim as optim
from torchsummary import summary           
import numpy as np
from PIL import ImageFilter
import numbers
import re
import math
import copy
import matplotlib.pyplot as plt
import gzip
import pickle
from torch.autograd import Variable
from scipy.ndimage import gaussian_filter

seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)


dataroot = "/content/drive/My Drive/DLStudio-1.1.0/Examples/data/"
image_size = [32,32]
path_saved_model = "./net2"
momentum = 0.9
learning_rate = 1e-5
epochs = 3
batch_size = 8
classes = ('rectangle','triangle','disk','oval','star')
debug_train = 1
debug_test = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_network_summary(net):
    print("\n\n\nprinting out the model:")
    print(net)
    print("\n\n\na summary of input/output for the model:")
    summary(net, (3,image_size[0],image_size[1]),-1, device = 'cpu')

def display_tensor_as_image(tensor, title=""):
    tensor_range = (torch.min(tensor).item(), torch.max(tensor).item())
    if tensor_range == (-1.0,1.0):
        print("\n\n\nimage un-normalization called")
        tensor = tensor/2.0 + 0.5     # unnormalize
    plt.figure(title)
    if tensor.shape[0] == 3 and len(tensor.shape) == 3:
        plt.imshow( tensor.numpy().transpose(1,2,0) )
    elif tensor.shape[0] == 1 and len(tensor.shape) == 3:
        tensor = tensor[0,:,:]
        plt.imshow( tensor.numpy(), cmap = 'gray' )
    elif len(tensor.shape) == 2:
        plt.imshow( tensor.numpy(), cmap = 'gray' )
    else:
        sys.exit("\n\n\nfrom 'display_tensor_as_image()': tensor for image is ill formed -- aborting")
    plt.show()


def save_model(model):
    torch.save(model.state_dict(), path_saved_model)

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

class DetectAndLocalize(nn.Module):             

    def __init__(self, dataserver_train=None, dataserver_test=None, dataset_file_train=None, dataset_file_test=None):
        self.dataserver_train = dataserver_train
        self.dataserver_test = dataserver_test

    class PurdueShapes5Dataset(torch.utils.data.Dataset):
        def __init__(self, dataset_file, noise = None, transform=None):
            super(DetectAndLocalize.PurdueShapes5Dataset, self).__init__()
            self.noise = noise
            if dataset_file == "PurdueShapes5-10000-train.gz":
                if os.path.exists("torch-saved-PurdueShapes5-10000-dataset.pt") and \
                        os.path.exists("torch-saved-PurdueShapes5-label-map-0.pt"):
                    print("\nLoading training data from the torch-saved archive")
                    self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset.pt")
                    self.label_map = torch.load("torch-saved-PurdueShapes5-label-map-0.pt")
                    # reverse the key-value pairs in the label dictionary:
                    self.class_labels = dict(map(reversed, self.label_map.items()))
                    self.transform = transform
                else:
                    print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                          """the dataset for this script. First time loading could take\n"""
                          """a minute or so.  Any subsequent attempts will only take\n"""
                          """a few seconds.\n\n\n""")
                    root_dir = dataroot
                    f = gzip.open(root_dir + dataset_file, 'rb')
                    dataset = f.read()
                    if sys.version_info[0] == 3:
                        self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                    else:
                        self.dataset, self.label_map = pickle.loads(dataset)
                    torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset.pt")
                    torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map-0.pt")
                    # reverse the key-value pairs in the label dictionary:
                    self.class_labels = dict(map(reversed, self.label_map.items()))
                    self.transform = transform
            elif dataset_file == "PurdueShapes5-10000-train-noise-20.gz":
                if os.path.exists("torch-saved-PurdueShapes5-10000-dataset-noise-20.pt") and \
                        os.path.exists("torch-saved-PurdueShapes5-label-map-1.pt"):
                    print("\nLoading training data from the torch-saved archive")
                    self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset-noise-20.pt")
                    self.label_map = torch.load("torch-saved-PurdueShapes5-label-map-1.pt")
                    # reverse the key-value pairs in the label dictionary:
                    self.class_labels = dict(map(reversed, self.label_map.items()))
                    self.transform = transform
                else:
                    print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                          """the dataset for this script. First time loading could take\n"""
                          """a minute or so.  Any subsequent attempts will only take\n"""
                          """a few seconds.\n\n\n""")
                    root_dir = dataroot
                    f = gzip.open(root_dir + dataset_file, 'rb')
                    dataset = f.read()
                    if sys.version_info[0] == 3:
                        self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                    else:
                        self.dataset, self.label_map = pickle.loads(dataset)
                    torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset-noise-20.pt")
                    torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map-1.pt")
                    # reverse the key-value pairs in the label dictionary:
                    self.class_labels = dict(map(reversed, self.label_map.items()))
                    self.transform = transform
            elif dataset_file == "PurdueShapes5-10000-train-noise-50.gz":
                if os.path.exists("torch-saved-PurdueShapes5-10000-dataset-noise-50.pt") and \
                        os.path.exists("torch-saved-PurdueShapes5-label-map-2.pt"):
                    print("\nLoading training data from the torch-saved archive")
                    self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset-noise-50.pt")
                    self.label_map = torch.load("torch-saved-PurdueShapes5-label-map-2.pt")
                    # reverse the key-value pairs in the label dictionary:
                    self.class_labels = dict(map(reversed, self.label_map.items()))
                    self.transform = transform
                else:
                    print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                          """the dataset for this script. First time loading could take\n"""
                          """a minute or so.  Any subsequent attempts will only take\n"""
                          """a few seconds.\n\n\n""")
                    root_dir = dataroot
                    f = gzip.open(root_dir + dataset_file, 'rb')
                    dataset = f.read()
                    if sys.version_info[0] == 3:
                        self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                    else:
                        self.dataset, self.label_map = pickle.loads(dataset)
                    torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset-noise-50.pt")
                    torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map-2.pt")
                    # reverse the key-value pairs in the label dictionary:
                    self.class_labels = dict(map(reversed, self.label_map.items()))
                    self.transform = transform
            elif dataset_file == "PurdueShapes5-10000-train-noise-80.gz":
                if os.path.exists("torch-saved-PurdueShapes5-10000-dataset-noise-80.pt") and \
                        os.path.exists("torch-saved-PurdueShapes5-label-map-3.pt"):
                    print("\nLoading training data from the torch-saved archive")
                    self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset-noise-80.pt")
                    self.label_map = torch.load("torch-saved-PurdueShapes5-label-map-3.pt")
                    # reverse the key-value pairs in the label dictionary:
                    self.class_labels = dict(map(reversed, self.label_map.items()))
                    self.transform = transform
                else:
                    print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                          """the dataset for this script. First time loading could take\n"""
                          """a minute or so.  Any subsequent attempts will only take\n"""
                          """a few seconds.\n\n\n""")
                    root_dir = dataroot
                    f = gzip.open(root_dir + dataset_file, 'rb')
                    dataset = f.read()
                    if sys.version_info[0] == 3:
                        self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                    else:
                        self.dataset, self.label_map = pickle.loads(dataset)
                    torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset-noise-80.pt")
                    torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map-3.pt")
                    # reverse the key-value pairs in the label dictionary:
                    self.class_labels = dict(map(reversed, self.label_map.items()))
                    self.transform = transform
            else:
                root_dir = dataroot
                f = gzip.open(root_dir + dataset_file, 'rb')
                dataset = f.read()
                if sys.version_info[0] == 3:
                    self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                else:
                    self.dataset, self.label_map = pickle.loads(dataset)
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                self.transform = transform


        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            r = np.array( self.dataset[idx][0] )
            g = np.array( self.dataset[idx][1] )
            b = np.array( self.dataset[idx][2] )
            R,G,B = r.reshape(32,32), g.reshape(32,32), b.reshape(32,32)
            im_tensor = torch.zeros(3,32,32, dtype=torch.float)
            im_tensor[0,:,:] = torch.from_numpy(R)
            im_tensor[1,:,:] = torch.from_numpy(G)
            im_tensor[2,:,:] = torch.from_numpy(B)
            bb_tensor = torch.tensor(self.dataset[idx][3], dtype=torch.float)
            sample = {'image' : im_tensor, 
                      'bbox' : bb_tensor,
                      'label' : self.dataset[idx][4],
                      'noise' : self.noise
                      }
            return sample

    def load_PurdueShapes5_dataset(self, dataserver_train, dataserver_test ):       

        self.train_dataloader = torch.utils.data.DataLoader(dataserver_train,
                           batch_size=batch_size,shuffle=True, num_workers=4)
        self.test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                           batch_size=batch_size,shuffle=False, num_workers=4)

#inspired from https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351

    def gauss_kernel(self,kernel_size=3, sigma=2, channels=3):
        x = torch.arange(kernel_size)
        x_grid = x.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    
        mean = (kernel_size - 1)/2.
        variance = sigma**2.
    
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                          torch.exp(
                              -torch.sum((xy_grid - mean)**2., dim=-1) /\
                              (2*variance)
                          )
    
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    
        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size,padding=1, groups=channels, bias=False)
    
        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
        
        return gaussian_filter
    

    class SkipBlock(nn.Module):
        def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
            super(DetectAndLocalize.SkipBlock, self).__init__()
            self.downsample = downsample
            self.skip_connections = skip_connections
            self.in_ch = in_ch
            self.out_ch = out_ch
            self.convo = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
            norm_layer = nn.BatchNorm2d
            self.bn = norm_layer(out_ch)
            if downsample:
                self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)

        def forward(self, x):
            identity = x                                     
            out = self.convo(x)                              
            out = self.bn(out)                              
            out = torch.nn.functional.relu(out)
            if self.in_ch == self.out_ch:
                out = self.convo(out)                              
                out = self.bn(out)                              
                out = torch.nn.functional.relu(out)
            if self.downsample:
                out = self.downsampler(out)
                identity = self.downsampler(identity)
            if self.skip_connections:
                if self.in_ch == self.out_ch:
                    out += identity                              
                else:
                    out[:,:self.in_ch,:,:] += identity
                    out[:,self.in_ch:,:,:] += identity
            return out
            
    class LOADnet2(nn.Module):
        def __init__(self, skip_connections=True, depth=32):
            super(DetectAndLocalize.LOADnet2, self).__init__()
            self.pool_count = 3
            self.depth = depth
            self.sigma= nn.Parameter(torch.tensor(1.0))
            self.filt = DetectAndLocalize.gauss_kernel(self, sigma = self.sigma)
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.skip641 = nn.ModuleList([DetectAndLocalize.SkipBlock(64, 64, 
                                                       skip_connections=skip_connections) for _ in range(self.depth // 4)])
            self.skip64ds = DetectAndLocalize.SkipBlock(64, 64, 
                                       downsample=True, skip_connections=skip_connections)
            self.skip642 = nn.ModuleList([DetectAndLocalize.SkipBlock(64, 64, 
                                                       skip_connections=skip_connections) for _ in range(self.depth // 4)])

            self.skip64to128 = DetectAndLocalize.SkipBlock(64, 128, 
                                                        skip_connections=skip_connections )
            self.skip1281 = nn.ModuleList([DetectAndLocalize.SkipBlock(128, 128, 
                                                         skip_connections=skip_connections) for _ in range(self.depth // 4)])
            self.skip128ds = DetectAndLocalize.SkipBlock(128,128,
                                        downsample=True, skip_connections=skip_connections)
            self.skip1282 = nn.ModuleList([DetectAndLocalize.SkipBlock(128, 128, 
                                                         skip_connections=skip_connections) for _ in range(self.depth // 4)])

            self.fc1 =  nn.Linear(128 * (32 // 2**self.pool_count)**2, 1000)
            self.fc2 =  nn.Linear(1000, 5) #for your shapes
            self.fc3 =  nn.Linear(1000, 4) #for your noise
            ##  for regression
            self.conv_seqn = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.fc_seqn = nn.Sequential(
                nn.Linear(16384, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 4)
            )

        def forward(self, x):

            x = self.filt(x)
            x = self.pool(torch.nn.functional.relu(self.conv(x)))          
            ## The labeling section:
            x1 = x.clone()
            #shape classifier
            for i in range(0, len(self.skip641)):
                x1 = self.skip641[i](x1)
            x1 = self.skip64ds(x1)
            for i in range(0, len(self.skip642)):
                x1 = self.skip642[i](x1) 
            x1 = self.skip64to128(x1)
            for i in range(0, len(self.skip1281)):
                x1 = self.skip1281[i](x1)                                               
            x1 = self.skip128ds(x1)                                               
            for i in range(0, len(self.skip1282)):
                x1 = self.skip1282[i](x1)                                               
            x1 = x1.view(-1, 128 * (32 // 2**self.pool_count)**2 )
            x1 = torch.nn.functional.relu(self.fc1(x1))
            x1 = self.fc2(x1)
            #bbox
            x2 = self.conv_seqn(x)
            x2 = self.conv_seqn(x2)
            x2 = x2.view(x.size(0), -1)
            x2 = self.fc_seqn(x2)
            x3 = x.clone() #to convolve the noise learnable parameters post filtering
            for i in range(0, len(self.skip641) // 2):
                x3 = self.skip641[i](x3)
            x3 = self.skip64ds(x3)
            for i in range(0, len(self.skip642) // 2):
                x3 = self.skip642[i](x3) 
            x3 = self.skip64to128(x3)
            for i in range(0, len(self.skip1281) // 2):
                x3 = self.skip1281[i](x3)                                               
            x3 = self.skip128ds(x3)                                               
            for i in range(0, len(self.skip1282) // 2):
                x3 = self.skip1282[i](x3)                                               
            x3 = x3.view(-1, 128 * (32 // 2**self.pool_count)**2 )
            x3 = torch.nn.functional.relu(self.fc1(x3))
            x3 = self.fc2(x3)

            return x1,x2, x3
            
        def get_num_correct(preds, labels):
            return preds.argmax(dim=1).eq(labels).sum().item()

    def run_code_for_training_with_CrossEntropy_and_MSE_Losses(self, net):        

        net = copy.deepcopy(net)
        net = net.to(device)
        criterion1 = nn.CrossEntropyLoss()
        criterion2 = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), 
                      lr=learning_rate, momentum=momentum)
        for epoch in range(epochs):  
            running_loss_labeling = 0.0
            running_loss_noise = 0.0
            running_loss_regression = 0.0       
            for i, data in enumerate(self.train_dataloader):
                inputs, bbox_gt, labels, noise = data['image'], data['bbox'], data['label'], data['noise']
                inputs = inputs.to(device)
                bbox_gt = bbox_gt.to(device)
                labels = labels.to(device)
                noise = noise.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                outputs_label = outputs[0]
                bbox_pred = outputs[1]
                noise_pred = outputs[2]
                loss_labeling = criterion1(outputs_label, labels)
                loss_labeling.backward(retain_graph=True)        
                loss_noise = criterion1(noise_pred, noise)
                loss_noise.backward(retain_graph=True)        
                loss_regression = criterion2(bbox_pred, bbox_gt)
                loss_regression.backward()
                optimizer.step()

                running_loss_labeling += loss_labeling.item()    
                running_loss_noise += loss_noise.item()                
                running_loss_regression += loss_regression.item()                
                if i % 500 == 499:    
                    avg_loss_labeling = running_loss_labeling / float(500)
                    avg_loss_regression = running_loss_regression / float(500)
                    avg_loss_noise = running_loss_noise / float(500)
                    print("\n[epoch:%d, iteration:%5d]  loss_labeling: %.3f  loss_regression: %.3f  " % (epoch + 1, i + 1, avg_loss_labeling, avg_loss_regression))
                    running_loss_labeling = 0.0
                    running_loss_regression = 0.0
                    running_loss_noise = 0.0
        print("\nFinished Training\n")
        save_model(net)

    def get_num_correct(preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def run_code_for_testing_detection_and_localization(self, net):
        net.load_state_dict(torch.load(path_saved_model))
        net = net.to(device)
        cmt1 = torch.zeros(5,5, dtype= torch.int64)
        cmt2 = torch.zeros(4,4, dtype= torch.int64)
        total_correct_shape = 0
        total_correct_noise = 0 
        for epoch in range(1):
            for i, data in enumerate(self.test_dataloader):
                inputs, bbox_gt, labels, noise = data['image'], data['bbox'], data['label'], data['noise']
                inputs = inputs.to(device)
                bbox_gt = bbox_gt.to(device)
                labels = labels.to(device)
                noise = noise.to(device)
                outputs = net(inputs)
                outputs_label = outputs[0]
                bbox_pred = outputs[1]
                noise_pred = outputs[2]
                total_correct_noise += get_num_correct(noise_pred, noise)
                total_correct_shape += get_num_correct(outputs_label, labels)
                stacked1 = torch.stack((labels, outputs_label.argmax(dim=1)), dim = 1)
                for p in stacked1:
                    tl, pl = p.tolist()
                    cmt1[tl, pl] = cmt1[tl, pl] + 1
                '''stacked2 = torch.stack((noise, noise_pred.argmax(dim=1)), dim = 1)
                for p in stacked2:
                    tl, pl = p.tolist()
                    cmt2[tl, pl] = cmt2[tl, pl] + 1'''
                    
        print("Dataset50 Classification Accuracy: %.3f%%" %(100*total_correct_shape / (i*batch_size)))
        print("Dataset50 Confusion Matrix")
        print(cmt1)
        
                
        '''print("Noise Classification Accuracy: %.3f%%" %(100*total_correct_noise / (1000)))
        #f.write("Classification Accuracy: %.3f%%" %(100*total_correct_noise / (i*batch_size)))
        print(cmt2)'''


detector = DetectAndLocalize()

dataserver_train1 = DetectAndLocalize.PurdueShapes5Dataset(
                                   dataset_file = "PurdueShapes5-10000-train.gz", noise = 0
                                                                      )
dataserver_test1 = DetectAndLocalize.PurdueShapes5Dataset(
                                   dataset_file = "PurdueShapes5-1000-test.gz",  noise = 0
                                                                  )
dataserver_train2 = DetectAndLocalize.PurdueShapes5Dataset(
                                   dataset_file = "PurdueShapes5-10000-train-noise-20.gz", noise = 1
                                                                      )
dataserver_test2 = DetectAndLocalize.PurdueShapes5Dataset(
                                   dataset_file = "PurdueShapes5-1000-test-noise-20.gz", noise = 1
                                                                  )
dataserver_train3 = DetectAndLocalize.PurdueShapes5Dataset(
                                   dataset_file = "PurdueShapes5-10000-train-noise-50.gz", noise = 2
                                                                      )
dataserver_test3 = DetectAndLocalize.PurdueShapes5Dataset(
                                   dataset_file = "PurdueShapes5-1000-test-noise-50.gz", noise = 2
                                                                  )
dataserver_train4 = DetectAndLocalize.PurdueShapes5Dataset(
                                   dataset_file = "PurdueShapes5-10000-train-noise-80.gz", noise = 3
                                                                      )
dataserver_test4 = DetectAndLocalize.PurdueShapes5Dataset(
                                   dataset_file = "PurdueShapes5-1000-test-noise-80.gz", noise = 3
                                                                  )

'''datasettr = [dataserver_train1, dataserver_train2, dataserver_train3, dataserver_train4] 
datasettst = [dataserver_test1, dataserver_test2, dataserver_test3, dataserver_test4] 

datasettrain = torch.utils.data.ConcatDataset(datasettr)
datasettest = torch.utils.data.ConcatDataset(datasettst)
print(len(datasettrain))

detector.dataserver_train = datasettrain
detector.dataserver_test = datasettest'''

detector.dataserver_train = dataserver_train3
detector.dataserver_test = dataserver_test3

detector.load_PurdueShapes5_dataset(detector.dataserver_train, detector.dataserver_test) #Just pass 50% noise level data

model = detector.LOADnet2(skip_connections=True, depth=32)
show_network_summary(model)
print(device)
detector.run_code_for_training_with_CrossEntropy_and_MSE_Losses(model)

detector.run_code_for_testing_detection_and_localization(model)

#The end! Thank you for this one-of-a-kind assignment. It was the most time consuming one in my grad life, and with high learning though my accuracies may not be that great in task 3! I learnt 3 techniques.  

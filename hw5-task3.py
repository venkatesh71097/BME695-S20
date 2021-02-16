##DLStudio removed code - testing - task 3 - proper as per Prof. Kak. There is going to be a flurry of my various trials of code to show my efforts on trying various styles for this task. Please bear with me!
## I'll append my run_code_for_training functions only as they were the ones where I kept playing excessively for 2-3 days.
#This task made me get stuck into coder's block, learn solving numerous errors in python, and the virtue of patience to keep debugging after errors in many lines. I've learning to solve more than 30-40 errors via this very program!  
#As always, credit to Prof. Kak for his very informative DLStudio code and my friends for making me conceptually understand stuffs when I get stuck!

'''Technique 1: Gives the best accuracy of all the three, but this doesn't follow what Prof. Kak and Nader have been reiterating us. But if allowed, I can produce this full network which gives an overall accuracy of 75-80%. Haven't tried for individual ones though. But I feel this can do very well. 
    for epoch in range(epochs):
        running_lossnoise = 0.0
        running_loss_labeling = 0.0
        running_loss_regression = 0.0
        for i, data in enumerate(train_data_loader):
            inputs, bbox_gt, noise, labels = data['image'], data['bbox'], data['noise'], data['label']
            gt_too_small = False
            inputs = inputs.to(device) #image
            bbox_gt = bbox_gt.to(device) #bbox 
            noise = noise.to(device) #noise
            labels = labels.to(device) #label
            #we need to first take noiseclassifier - so inputs[0] and labels[0]
            optimizer1.zero_grad()
            output_net1 = net1(inputs) Here, I had passed my noise classifier to inputs, got the predicted value for every image in the batch, modified the input, again passed the input to output_net1 to
            mitigate inplace operator error, and then pass the filtered inputs now into your LOADnet. 
            for j in range(batch_size):
                pred = torch.argmax(output_net1[j])
                if pred == 0: 
                    inputs[j] = torch.from_numpy((inputs[j].cpu().numpy())).to(device)
                if pred == 1: 
                    inputs[j] = torch.from_numpy(gaussian_filter(inputs[j].cpu().numpy(),sigma = 0.5)).to(device) 
                if pred == 2: 
                    inputs[j] = torch.from_numpy(gaussian_filter(inputs[j].cpu().numpy(),sigma = 1)).to(device)
                if pred == 3: 
                    inputs[j] = torch.from_numpy(gaussian_filter(inputs[j].cpu().numpy(),sigma = 1.2)).to(device)
            
            output_net1 = net1(inputs)
            lossnoise = criterion0(output_net1, noise)
            lossnoise.backward()
            optimizer1.step()
            running_lossnoise = running_lossnoise + lossnoise.item()
            optimizer2.zero_grad()
            output_net2 = net2(inputs)
            outputs_label = output_net2[0]
            bbox_pred = output_net2[1]

            loss_labeling = criterion1(outputs_label, labels)
            loss_labeling.backward(retain_graph=True)        
            loss_regression = criterion2(bbox_pred, bbox_gt)
            loss_regression.backward()
            optimizer2.step()
            running_loss_labeling += loss_labeling.item()    
            running_loss_regression += loss_regression.item()                
'''

'''Technique 2: I used this following block both in the training and testing part: I wanted to train both my LOADnet and ClassifierNet at the same time. It was running very slowly that I couldn't see any result till 30-40 minutes.
I'd like to explore this particular architecture as I feel, individually training and testing both networks can give a foolproof network. This costed me more than half a day that I was disappointed to see it not working
            output_net1 = net(inputs) #noise
            a = torch.zeros(batch_size, 5)
            b = torch.zeros(batch_size, 4)
            output_net2 = [a,b] #for classification and regression as they're the two outputs. 
            optimizer2.zero_grad()
            for j in range(batch_size):
                pred = torch.argmax(output_net1[j])
                output_net2 = net2(inputs[j].unsqueeze(0))
                print("works")
            if pred == 0: 
                output_net2[0][j] = net2(inputs[j].unsqueeze(0))[0]
                output_net2[1][j] = net2(inputs[j].unsqueeze(0))[1]
            if pred == 1: 
                output_net2[0][j] = net3(inputs[j].unsqueeze(0))[0]
                output_net2[1][j] = net3(inputs[j].unsqueeze(0))[1]
            if pred == 2: 
                output_net2[0][j] = net4(inputs[j].unsqueeze(0))[0]
                output_net2[1][j] = net4(inputs[j].unsqueeze(0))[1]
            if pred == 3: 
                output_net2[0][j] = net5(inputs[j].unsqueeze(0))[0]
                output_net2[1][j] = net5(inputs[j].unsqueeze(0))[1]
            
'''

'''The third and the final technique that I used, gave me not-so-good accuracy but, it looked almost close to what Prof. Kak and Nader seem to expect. I just trained the noise classifer
and in the testing, I invoke the pretrained models for the different noise levels (net1, net2, net3, net4) to choose their respective unique weights to run on the inputs. Unfortuantely, it gave
pretty bad results for 80% noise data, with accuracy of around 34%, but for others it performed somewhat decently'''

##Also, I had used same architecture of LOADnet2 to my ClassifierNet as it gave me close to 99-100 percent accuracy even in the very first epoch, and is performing exceptionally well, with the lowest individual noise classification for I guess, 50% noise data, is 98.8 %. 

##To summarize, I used 3 different techniques for this task, which took around 2-2.5 days for me to analyze which matches Prof's requirements. Sorry to hand in the lower accuracy one. I'll try working on it later to improve it further. 
## But, I'm very sure that these are certain styles which I personally find to be lucid, elementary and unique. Hence, it pumped me up to try such styles, though I might have not been entirely successful. 

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
momentum = 0.9
learning_rate = 1e-5
epochs = 1
batch_size = 8
classes = ('rectangle','triangle','disk','oval','star')
debug_train = 1
debug_test = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.autograd.set_detect_anomaly(True)
total_correct_noise = 0
total_correct_shape = 0


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#I feel this reduces accuracy when it is removed. Something is wrong with this. 
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


def save_model(model, name):
    torch.save(model.state_dict(), name)


class SkipBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
        super(SkipBlock, self).__init__()
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

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()
#My noise classifier network. Same as that of LOADnet architecture, with some minute modifications here and there to make it denser. 
class ClassifierNet(nn.Module):
    def __init__(self, skip_connections=True, depth = 32):
        super(ClassifierNet, self).__init__()
        self.pool_count = 3
        self.depth = depth // 2
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.total_correct_noise = 0
        self.skip641 = nn.ModuleList([SkipBlock(64, 64, 
                                                    skip_connections=skip_connections) for _ in range(self.depth // 4)])
        self.skip64ds = SkipBlock(64, 64, 
                                    downsample=True, skip_connections=skip_connections)
        self.skip642 = nn.ModuleList([SkipBlock(64, 64, 
                                                    skip_connections=skip_connections) for _ in range(self.depth // 4)])

        self.skip64to128 = SkipBlock(64, 128, 
                                                    skip_connections=skip_connections )
        self.skip1281 = nn.ModuleList([SkipBlock(128, 128, 
                                                      skip_connections=skip_connections) for _ in range(self.depth // 4)])
        self.skip128ds = SkipBlock(128,128,
                                    downsample=True, skip_connections=skip_connections)
        self.skip1282 = nn.ModuleList([SkipBlock(128, 128, 
                                                      skip_connections=skip_connections) for _ in range(self.depth // 4)])

        self.fc1 =  nn.Linear(128 * (32 // 2**self.pool_count)**2, 1000)
        self.fc2 =  nn.Linear(1000, 4)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv(x)))          
        ## The labeling section:
        x1 = x.clone()
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
        return x1  

    def get_num_correct(preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

########################################################################################
###################  Start Definition of Inner Class DetectAndLocalize  ################

class DetectAndLocalize(nn.Module):             

    def __init__(self, dataserver_train=None, dataserver_test=None, dataset_file_train=None, dataset_file_test=None):
        self.dataserver_train = dataserver_train
        self.dataserver_test = dataserver_test
        super(DetectAndLocalize, self).__init__()
    #Removed the train_or_test parameter as I didn't need it  
    class PurdueShapes5Dataset(torch.utils.data.Dataset):
        def __init__(self, dataset_file, noise, transform=None):
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
                      'bbox'  : bb_tensor,
                      'label' : self.dataset[idx][4],
                      'noise' : self.noise
                      } #very simple way rather than declaring a dictionary variable, updating with noise values is to pass noise as an argument to fetch the dataset and use self.noise to append to a dictionary for each image
            return sample

    def load_PurdueShapes5_dataset(self, dataserver_train, dataserver_test ):       

        self.train_dataloader = torch.utils.data.DataLoader(dataserver_train,
                           batch_size=batch_size,shuffle=True, num_workers=4)
        self.test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                           batch_size=batch_size,shuffle=False, num_workers=4)
        return self.train_dataloader, self.test_dataloader #made it returnable to use it universally without the object instance. 

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
        def __init__(self, skip_connections=True, depth = 32):
            super(DetectAndLocalize.LOADnet2, self).__init__()
            self.pool_count = 3
            self.depth = depth // 2
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
            self.fc2 =  nn.Linear(1000, 5)

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
            x = self.pool(torch.nn.functional.relu(self.conv(x)))          
            ## The labeling section:
            x1 = x.clone()
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
            ## The Bounding Box regression:
            x2 = self.conv_seqn(x)
            x2 = self.conv_seqn(x2)
            # flatten
            x2 = x2.view(x.size(0), -1)
            x2 = self.fc_seqn(x2)
            return x1, x2            
    
def run_code_for_training(net1, train_data_loader, epochs): 
    net1 = net1.to(device)
#As I said, this function is just for training my noise classifier. 
    criterion0 = nn.CrossEntropyLoss()

    optimizer1 = torch.optim.SGD(net1.parameters(), lr=1e-4, momentum=0.9)

    for epoch in range(epochs):
        running_lossnoise = 0.0
        running_loss_labeling = 0.0
        running_loss_regression = 0.0
        for i, data in enumerate(train_data_loader):
            inputs, bbox_gt, noise, labels = data['image'], data['bbox'], data['noise'], data['label']
            inputs = inputs.to(device) #image
            bbox_gt = bbox_gt.to(device) #bbox 
            noise = noise.to(device) #noise
            labels = labels.to(device) #label
            #we need to first take noiseclassifier - so inputs[0] and labels[0]
            optimizer1.zero_grad()
            output_net1 = net1(inputs) 
            lossnoise = criterion0(output_net1, noise)
            lossnoise.backward()
            optimizer1.step()
            running_lossnoise = running_lossnoise + lossnoise.item()
            if i % 500 == 499:    
                avg_loss_noise = running_lossnoise / float(500)
                #avg_loss_labeling = running_loss_labeling / float(500)
                #avg_loss_regression = running_loss_regression / float(500)
                print("\n[epoch:%d, iteration:%5d]  loss_noise: %.3f " % (epoch + 1, i + 1, avg_loss_noise))
                running_lossnoise = 0.0   

    print("\nFinished Training\n")
    save_model(net1, path_saved_model1)


def run_code_for_testing(net1, path_saved_model1, test_data_loader):
    #Here comes my idea. Why not call the pretrained weights and assign it to our LOADnet2 model so that we can make use of 4 different instances with different weights to train on our data being passed via dataloader. I personally find this as a cool idea. 
    net1.load_state_dict(torch.load(path_saved_model1)) #noise network
    net2 = DetectAndLocalize.LOADnet2(skip_connections=True, depth = 32) 
    net3 = DetectAndLocalize.LOADnet2(skip_connections=True, depth = 32)
    net4 = DetectAndLocalize.LOADnet2(skip_connections=True, depth = 32)
    net5 = DetectAndLocalize.LOADnet2(skip_connections=True, depth = 32)
    #These networks are the values got from task 2, which are trained there only, and directly loaded here. Training is bypassed here as my 'Technique 2' did nothing productive other than crashing my browser
    net2.load_state_dict(torch.load("./net2"), strict = False) #make strict false inorder to avoid 'key' issues which doesn't load your parameters 
    net3.load_state_dict(torch.load("./net3"), strict = False)
    net4.load_state_dict(torch.load("./net4"), strict = False)
    net5.load_state_dict(torch.load("./net5"), strict = False)
    
    net1 = net1.to(device)
    net2 = copy.deepcopy(net2) #still don't have clarity on when to use deepcopy (Just to create multiple copies so that original copies don't get affected. But what else other than this?)
    net2 = net2.to(device)
    net3 = net3.to(device)
    net4 = net4.to(device)
    net5 = net5.to(device)
    print("It works") #Checkpoints to success :p 
    cmt1 = torch.zeros(4,4, dtype= torch.int64)
    cmt2 = torch.zeros(5,5, dtype= torch.int64)
    total_correct_shape = 0
    total_correct_noise = 0
    for i, data in enumerate(test_data_loader):
        inputs, bbox_gt, noise, labels = data['image'], data['bbox'], data['noise'], data['label']
        inputs = inputs.to(device) #image
        bbox_gt = bbox_gt.to(device) #bbox 
        noise = noise.to(device) #noise
        labels = labels.to(device) #label
        output_net1 = net1(inputs) 
        print(noise)
        total_correct_noise += get_num_correct(output_net1, noise)
        a = torch.zeros(batch_size, 5) #Imperative to initialize your values so that you don't get an immutable tuple. You need a list which can be modified. Or else, you will have to face error of referencing before assignment
        b = torch.zeros(batch_size, 4)
        output_net2 = [a,b]
        for j in range(0, batch_size):
            pred = torch.argmax(output_net1[j])
            print(pred)
            if pred == 0: 
                inputs[j] = torch.from_numpy((inputs[j].cpu().numpy())).to(device)
                output_net2[0][j] = net2(inputs[j].unsqueeze(0))[0] #unsqueeze makes 3d tensor to 4d. net2 gives 2 outputs. Index 0 gives first tensor, 1 gives 2nd tensor. Don't make it like a tuple. 
                output_net2[1][j] = net2(inputs[j].unsqueeze(0))[1]
            if pred == 1: 
                inputs[j] = torch.from_numpy(gaussian_filter(inputs[j].cpu().numpy(),sigma = 0.83, truncate = 3)).to(device) #7x7 filter. Using truncate*sigma formula 
                output_net2[0][j] = net3(inputs[j].unsqueeze(0))[0]
                output_net2[1][j] = net3(inputs[j].unsqueeze(0))[1]
            if pred == 2: 
                inputs[j] = torch.from_numpy(gaussian_filter(inputs[j].cpu().numpy(),sigma = 0.75, truncate = 2)).to(device) #5x5 filter
                output_net2[0][j] = net4(inputs[j].unsqueeze(0))[0]
                output_net2[1][j] = net4(inputs[j].unsqueeze(0))[1]
            if pred == 3: 
                inputs[j] = torch.from_numpy(gaussian_filter(inputs[j].cpu().numpy(),sigma = 0.5, truncate = 1)).to(device) #3x3 filter. My logic was, as noise increases, it makes sense to lower the kernel size so as to fetch better results within a smaller region rather than convolve with a bigger filter and collapse the convolution with the noise paramters
                output_net2[0][j] = net5(inputs[j].unsqueeze(0))[0]
                output_net2[1][j] = net5(inputs[j].unsqueeze(0))[1]

        #output_net1 = net1(inputs) #removes inplace error. Not needed here in test as we aren't passing backprop. Needed in training if you use my first two techniques mentioned to mitigate inplace operator error. 
        outputs_label = output_net2[0]
        bbox_pred = output_net2[1]
        outputs_label.to(device)
        bbox_pred.to(device)
        print(outputs_label)

        outputs_label = outputs_label.detach().cpu()
        labels = labels.detach().cpu()
        outputs_label.to(device)
        labels.to(device)
        total_correct_shape += get_num_correct(outputs_label, labels)
        stacked1 = torch.stack((noise, output_net1.argmax(dim=1)), dim = 1)
        for p in stacked1:
            tl, pl = p.tolist()
            cmt1[tl, pl] = cmt1[tl, pl] + 1
        stacked2 = torch.stack((labels, outputs_label.argmax(dim=1)), dim = 1)
        for p in stacked2:
            tl, pl = p.tolist()
            cmt2[tl, pl] = cmt2[tl, pl] + 1

    print("Noise Classification Accuracy: %.3f \n" % (total_correct_noise * 100 / (i*batch_size)))
    print("Noise Confusion Matrix:\n")
    print(cmt1)
    print("\n")
    print("Dataset0 Classification Accuracy: %.3f \n" % (total_correct_shape * 100 / (i*batch_size)))
    print("Dataset0 Confusion Matrix:\n")
    print(cmt2)
    print("\n")



#loading and training

detector = DetectAndLocalize()

dataserver_train1 = DetectAndLocalize.PurdueShapes5Dataset(
                                   dataset_file = "PurdueShapes5-10000-train.gz", noise = 0
                                                                      )
dataserver_test1 = DetectAndLocalize.PurdueShapes5Dataset(
                                   dataset_file = "PurdueShapes5-1000-test.gz", noise = 0
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
print(type(dataserver_test4))
datasettr = [dataserver_train1, dataserver_train2, dataserver_train3, dataserver_train4] 
datasettst = [dataserver_test1, dataserver_test2, dataserver_test3, dataserver_test4] 

dataserver_train = torch.utils.data.ConcatDataset(datasettr)
dataserver_test = torch.utils.data.ConcatDataset(datasettst)
print(len(dataserver_train))

path_saved_model1 = "./net1"

detector.dataserver_train = dataserver_train
detector.dataserver_test = dataserver_test

train, test = detector.load_PurdueShapes5_dataset(detector.dataserver_train, detector.dataserver_test)
noisemodel = ClassifierNet(skip_connections=True, depth = 32)

'''show_network_summary(noisemodel)
show_network_summary(loadmodel)'''

print(device)
run_code_for_training(noisemodel, train, epochs)

#test block #I pass noise as a paramter here rather than using dict.update(). Makes it easier for me to append to my 'sample' dictionary. 
dataserver_test4 = DetectAndLocalize.PurdueShapes5Dataset(
                                   dataset_file = "PurdueShapes5-1000-test-noise-80.gz",noise=3
                                                                  )
#shuffling the dataset, thought they are testdata of individual noise labels does some good in generalizing output values. 
test1 = torch.utils.data.DataLoader(dataserver_test1,
                           batch_size=batch_size,shuffle=True, num_workers=4)
test2 = torch.utils.data.DataLoader(dataserver_test2,
                           batch_size=batch_size,shuffle=True, num_workers=4)
test3 = torch.utils.data.DataLoader(dataserver_test3,
                           batch_size=batch_size,shuffle=True, num_workers=4)
test4 = torch.utils.data.DataLoader(dataserver_test4,
                           batch_size=batch_size,shuffle=True, num_workers=4)

#Due to paucity of time, I'm just running everything one by one by changing the last parameter in run_code_for_testing and appending to my output.txt. To reiterate, we train the noise classifier with all data from all noise levels, and test one by one for every dataset. 
run_code_for_testing( noisemodel, path_saved_model1, test3)

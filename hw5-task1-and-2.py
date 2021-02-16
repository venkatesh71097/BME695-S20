##DLStudio removed code - task 2 

#!/usr/bin/env python
# editing current file
##  object_detection_and_localization.py
#Thanks to Prof. Kak for his insights using DLStudio. It was way helpful, and is really helpful in understanding concepts.
#Thanks to my friends, Manu, Praneet, Gaurav, Suyash, Akhil for guiding me then and there whenever I'm stuck in issues.
#I had been going crazy over the last one week with this problem statement as it was a very tricky one. I had come up with various solutions for task 2 and 3.
#Here, in task 2, I had implemented a static gaussian filter with 7x7 for 20% noise, 5x5 for 50% noise, 3x3 for 80% noise and got decent accuracies (Values keep changing now and then with +/- 5 percent deviation). 
#I had also extensively referred to stackoverflow, GitHub issues column, PyTorch forums to solve my never-ending errors. Though my accuracy results might not be promising, I promise to show
#you some of my earnest efforts in making something really unique for tasks 3 especially.

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

#parameters to initialize. DLS_new is the DLStudio-1.1.0. 10^-5 learning rate helps in mitigating the NaNs to an extent. 

dataroot = "/content/drive/My Drive/DLS_new/Examples/data/"
image_size = [32,32]
path_saved_model = "./net4"
momentum = 0.9
learning_rate = 1e-5
epochs = 5
batch_size = 8
classes = ('rectangle','triangle','disk','oval','star')
debug_train = 1
debug_test = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#I had used this initially to print images 
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#It runs on cpu and when in GPU, I face errors in displaying output. 
def show_network_summary(net):
    print("\n\n\nprinting out the model:")
    print(net)
    print("\n\n\na summary of input/output for the model:")
    summary(net, (3,image_size[0],image_size[1]),-1, device = 'cpu')

#Used earlier to visualize outputs
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

#One stop place to save my models. Comes in handy for task 3. 
def save_model(model):
    torch.save(model.state_dict(), path_saved_model)


########################################################################################
###################  Start Definition of Inner Class DetectAndLocalize  ################

class DetectAndLocalize(nn.Module):             

    def __init__(self, dataserver_train=None, dataserver_test=None, dataset_file_train=None, dataset_file_test=None):
        self.dataserver_train = dataserver_train
        self.dataserver_test = dataserver_test

    class PurdueShapes5Dataset(torch.utils.data.Dataset):
        def __init__(self, dataset_file, transform=None):
            super(DetectAndLocalize.PurdueShapes5Dataset, self).__init__()
            if dataset_file == "PurdueShapes5-10000-train.gz":
                if os.path.exists("torch-saved-PurdueShapes5-10000-dataset.pt") and \
                        os.path.exists("torch-saved-PurdueShapes5-label-map-0.pt"):
                    print("\nLoading training data from the torch-saved archive")
                    self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset.pt")
                    self.label_map = torch.load("torch-saved-PurdueShapes5-label-map-0.pt")
                    # reverse the key-value pairs in the label dictionary: (Courtesy: Prof Kak)
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
            im_tensor[0,:,:] = torch.from_numpy(gaussian_filter(R, sigma = 0.75, truncate = 2)) #(w-1) / 2 - 0.5 = truncate * sigma (This is for 5x5 filter) - Courtesy: Github Repo of Gaussian filter code of scipy
            im_tensor[1,:,:] = torch.from_numpy(gaussian_filter(G, sigma = 0.75, truncate = 2)) 
            im_tensor[2,:,:] = torch.from_numpy(gaussian_filter(B, sigma = 0.75, truncate = 2))
            bb_tensor = torch.tensor(self.dataset[idx][3], dtype=torch.float)
            sample = {'image' : im_tensor, 
                      'bbox' : bb_tensor,
                      'label' : self.dataset[idx][4]
                      }
            return sample

    def load_PurdueShapes5_dataset(self, dataserver_train, dataserver_test ):       

        self.train_dataloader = torch.utils.data.DataLoader(dataserver_train,
                           batch_size=batch_size,shuffle=True, num_workers=4)
        self.test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                           batch_size=batch_size,shuffle=False, num_workers=4)
        
    #No changes in the skipblock as making it different made the output go way too weird. 
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

#Exactly Prof. Kak's LOADNet, just that myself and my friends spotted that the earlier SkipBlocks weren't growing denser. So, we decided to use nn.ModuleList to grow it layer by layer. 
    class LOADnet2(nn.Module):
        def __init__(self, skip_connections=True, depth=32):
            super(DetectAndLocalize.LOADnet2, self).__init__()
            self.pool_count = 3
            self.depth = depth
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
            self.fc2 =  nn.Linear(1000, 5) #Classification outputs 5
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
            )#Regression outputs 4

        def forward(self, x):
            #x = self.gaussianBlur2(x)
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
            return x1,x2

    def run_code_for_training_with_CrossEntropy_and_MSE_Losses(self, net):        
        net = copy.deepcopy(net)
        net = net.to(device)
        criterion1 = nn.CrossEntropyLoss()
        criterion2 = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), 
                      lr=learning_rate, momentum=momentum)
        for epoch in range(epochs):  
            running_loss_labeling = 0.0
            running_loss_regression = 0.0       
            for i, data in enumerate(self.train_dataloader):
                gt_too_small = False
                inputs, bbox_gt, labels = data['image'], data['bbox'], data['label']
                inputs = inputs.to(device)
                labels = labels.to(device)
                bbox_gt = bbox_gt.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                outputs_label = outputs[0]
                bbox_pred = outputs[1]

                loss_labeling = criterion1(outputs_label, labels)
                loss_labeling.backward(retain_graph=True)        
                loss_regression = criterion2(bbox_pred, bbox_gt)
                loss_regression.backward()
                optimizer.step()
                running_loss_labeling += loss_labeling.item()    
                running_loss_regression += loss_regression.item()                
                if i % 500 == 499:    
                    avg_loss_labeling = running_loss_labeling / float(500)
                    avg_loss_regression = running_loss_regression / float(500)
                    print("\n[epoch:%d, iteration:%5d]  loss_labeling: %.3f  loss_regression: %.3f  " % (epoch + 1, i + 1, avg_loss_labeling, avg_loss_regression))
                    running_loss_labeling = 0.0
                    running_loss_regression = 0.0

        print("\nFinished Training\n")
        save_model(net)


    def run_code_for_testing_detection_and_localization(self, net):
        net.load_state_dict(torch.load(path_saved_model))
        net.load_state_dict(torch.load(path_saved_model))

        correct = 0
        total = 0
        confusion_matrix = torch.zeros(len(self.dataserver_train.class_labels), 
                                       len(self.dataserver_train.class_labels))
        class_correct = [0] * len(self.dataserver_train.class_labels)
        class_total = [0] * len(self.dataserver_train.class_labels)
        with torch.no_grad():
            for i, data in enumerate(self.test_dataloader):
                images, bounding_box, labels = data['image'], data['bbox'], data['label']
                labels = labels.tolist()
                outputs = net(images)
                outputs_label = outputs[0]
                outputs_regression = outputs[1]
                outputs_regression[outputs_regression < 0] = 0
                outputs_regression[outputs_regression > 31] = 31
                outputs_regression[torch.isnan(outputs_regression)] = 0
                output_bb = outputs_regression.tolist()
                _, predicted = torch.max(outputs_label.data, 1)
                predicted = predicted.tolist()
                for label,prediction in zip(labels,predicted):
                    confusion_matrix[label][prediction] += 1
                total += len(labels)
                correct +=  [predicted[ele] == labels[ele] for ele in range(len(predicted))].count(True)
                comp = [predicted[ele] == labels[ele] for ele in range(len(predicted))]
                for j in range(batch_size):
                    label = labels[j]
                    class_correct[label] += comp[j]
                    class_total[label] += 1
        print("\n")
        #I used Prof. Kak's formatting style for this task and used my train and test functions where I input my confusion matrices to tensors. I used the output from here to visually align my output in notepad. I had not writting my train test functions in this task to use his style of printing to my other 2 tasks. 
        for j in range(len(self.dataserver_train.class_labels)):
            print('Prediction accuracy for %5s : %2d %%' % (
          self.dataserver_train.class_labels[j], 100 * class_correct[j] / class_total[j]))
        print("\n\n\n Classification Accuracy: %d %%" % 
                                                               (100 * correct / float(total)))
        print("\n\n Confusion matrix:\n")
        out_str = "                "
        for j in range(len(self.dataserver_train.class_labels)):  
                             out_str +=  "%15s" % self.dataserver_train.class_labels[j]   
        print(out_str + "\n")
        for i,label in enumerate(self.dataserver_train.class_labels):
            out_percents = [100 * confusion_matrix[i,j] / float(class_total[i]) 
                             for j in range(len(self.dataserver_train.class_labels))]
            out_percents = ["%.2f" % item.item() for item in out_percents]
            out_str = "%12s:  " % self.dataserver_train.class_labels[i]
            for j in range(len(self.dataserver_train.class_labels)): 
                                                   out_str +=  "%15s" % out_percents[j]
            print(out_str)


#We instantiate objects for every train and test sample
detector = DetectAndLocalize()

dataserver_train1 = DetectAndLocalize.PurdueShapes5Dataset(
                                   dataset_file = "PurdueShapes5-10000-train.gz"
                                                                      )
dataserver_test1 = DetectAndLocalize.PurdueShapes5Dataset(
                                   dataset_file = "PurdueShapes5-1000-test.gz"
                                                                  )

dataserver_train2 = DetectAndLocalize.PurdueShapes5Dataset(
                                   dataset_file = "PurdueShapes5-10000-train-noise-20.gz"
                                                                      )
dataserver_test2 = DetectAndLocalize.PurdueShapes5Dataset(
                                   dataset_file = "PurdueShapes5-1000-test-noise-20.gz"
                                                                  )

dataserver_train3 = DetectAndLocalize.PurdueShapes5Dataset(
                                   dataset_file = "PurdueShapes5-10000-train-noise-50.gz"
                                                                      )
dataserver_test3 = DetectAndLocalize.PurdueShapes5Dataset(
                                   dataset_file = "PurdueShapes5-1000-test-noise-50.gz"
                                                                  )

dataserver_train4 = DetectAndLocalize.PurdueShapes5Dataset(
                                   dataset_file = "PurdueShapes5-10000-train-noise-80.gz"
                                                                      )
dataserver_test4 = DetectAndLocalize.PurdueShapes5Dataset(
                                   dataset_file = "PurdueShapes5-1000-test-noise-80.gz"
                                                                  )

'''datasettr = [dataserver_train1, dataserver_train2, dataserver_train3, dataserver_train4] 
datasettst = [dataserver_test1, dataserver_test2, dataserver_test3, dataserver_test4] 

datasettrain = torch.utils.data.ConcatDataset(datasettr)
datasettest = torch.utils.data.ConcatDataset(datasettst)
print(len(datasettrain))

detector.dataserver_train = datasettrain
detector.dataserver_test = datasettest'''
# We ain't using the entire dataset for training. We use each noise level's dataset for training and the test part of the corresponding level for testing.

detector.dataserver_train = dataserver_train1
detector.dataserver_test = dataserver_test1
detector.load_PurdueShapes5_dataset(detector.dataserver_train, detector.dataserver_test)
model = detector.LOADnet2(skip_connections=True, depth=32)
detector.run_code_for_testing_detection_and_localization(model)

detector.dataserver_train = dataserver_train2
detector.dataserver_test = dataserver_test2
detector.load_PurdueShapes5_dataset(detector.dataserver_train, detector.dataserver_test)
model = detector.LOADnet2(skip_connections=True, depth=32)
detector.run_code_for_testing_detection_and_localization(model)

detector.dataserver_train = dataserver_train3
detector.dataserver_test = dataserver_test3
detector.load_PurdueShapes5_dataset(detector.dataserver_train, detector.dataserver_test)
model = detector.LOADnet2(skip_connections=True, depth=32)
detector.run_code_for_testing_detection_and_localization(model)

detector.dataserver_train = dataserver_train4
detector.dataserver_test = dataserver_test4
detector.load_PurdueShapes5_dataset(detector.dataserver_train, detector.dataserver_test)
model = detector.LOADnet2(skip_connections=True, depth=32)
detector.run_code_for_testing_detection_and_localization(model)

#Task1, 2, 3 - Final copy

'''Credits: Prof. Kak, DLStudio-1.1.3. Thanks to my friends Suyash, Praneet, Gaurav, Manu, Akhil for helping me out whwenever I'm in need of
help. I had done tasks 1, 2 and 3 using a very simple logic which worked for me, which I'll explain as we go through the code. '''

'''P.S.  I have run dataset-200 for Task 2, which runs faster (also produces the better result for 200 out of all 3). I can attach the results if needed.
Since it takes a lot of time to run, and also since debugging is tough, I got permission from him to run 2 models for 40 and 1 for 200. '''

'''I attach the results of 200 dataset of task 1. I didn't generate for 3 as it was super long : 

Task 1: 

[epoch:1  iter:7400  elapsed_time:7821 secs]     loss: 0.705

                      predicted negative    predicted positive

true negative:              17.837            82.163
true positive:               13.98             86.02
'''

'''Technique adopted:

Task 1: I had used GRUnet, and had passed integer hotted reviews rather than one hot encoded values. This yielded me better performance and accuracy. I think that negative Log loss function will
need a proper real value to predict properly, and not 0-1 for which there is Cross Entropy loss function. So, we pass the word index values directly to the hotvec variable. 

Task 2: I had used Textnet initially to see that a learning rate of 10^-6 isn't giving NANs and gives proper accuracy. To make the model more manageable, that is, making it sophisticated
with minimal complications compared to our GRUnet, we use TextnetOrder 2. This network is an attempt at creating a more elaborate relationship between the hidden state at time t and its
value at time t-1. Prof. Kak asked us to include sigmoid function to make it a gated function, and add inner state at t-1 also to be stored in cell as Linear combination with hidden state.
To be very honest, I found that it didn't yield any better accuracy. Infact, it was kind of performing slightly lower for me than Textnet if I'm not wrong. 

Task 3: Batching actually reduced the quality of the results surprisingly for me. I guess, the padding technique that I adopted is very inefficient as it takes more time to go through
unnecessary lengths more than its usual length. I didn't use truncating to a particular length as I was not fully convinced about the logic as I felt it'll let go off certain useful
parts to get classified into positive or negative. I thought of taking the length of every review in an array and select the median value, but all that was time consuming and with
exams this week, it was very difficult for me to think and work on those.'''


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
import time 
from torch.nn.utils.rnn import pad_sequence

seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)

dataroot = "/content/drive/My Drive/DLStudio-1.1.3/Examples/data/"
path_saved_model = "./net2"
momentum = 0.9
learning_rate = 0.000001 #Textnet doesn't work at 1e-3
epochs = 1 
batch_size = 1 #general batch size for tasks 1 and 2. 
classes = ('negative','positive')
debug_train = 1
debug_test = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def show_network_summary(self, net):
    print("\n\n\nprinting out the model:")
    print(net)
    print("\n\n\na summary of input/output for the model:")
    summary(net, (3,self.image_size[0],self.image_size[1]),-1, device='cpu')

def save_model(self, model):
    torch.save(model.state_dict(), self.path_saved_model)

#Inspiration from: https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html
#Used collate_fn argument from DataLoader function of torch.utils
def pad_collate(batch):
    review = []
    for x in batch:
        review.append(x['review'])
    review_pad = torch.nn.utils.rnn.pad_sequence(review, batch_first=True, padding_value=0)
    for i,x in enumerate(batch):
        x['review'] = review_pad[i]
    return batch
    
#Outer textclassification class
class TextClassification(nn.Module):             
    def __init__(self, dataserver_train=None, dataserver_test=None, dataset_file_train=None, dataset_file_test=None):
        #super(TextClassification, self).__init__()
        self.dataserver_train = dataserver_train
        self.dataserver_test = dataserver_test

    class SentimentAnalysisDataset(torch.utils.data.Dataset):
        """ For my reference... 
              sentiment_dataset_train_200.tar.gz        vocab_size = 43,285
              sentiment_dataset_test_200.tar.gz  

              sentiment_dataset_train_40.tar.gz         vocab_size = 17,001
              sentiment_dataset_test_40.tar.gz    

              sentiment_dataset_train_3.tar.gz          vocab_size = 3,402
              sentiment_dataset_test_3.tar.gz    
        """

        def __init__(self, train_or_test, dataset_file):
            super(TextClassification.SentimentAnalysisDataset, self).__init__()
            self.train_or_test = train_or_test
            root_dir = dataroot
            f = gzip.open(root_dir + dataset_file, 'rb')
            dataset = f.read()
            if train_or_test is 'train':
                if sys.version_info[0] == 3:
                    self.positive_reviews_train, self.negative_reviews_train, self.vocab = pickle.loads(dataset, encoding='latin1')
                else:
                    self.positive_reviews_train, self.negative_reviews_train, self.vocab = pickle.loads(dataset)

                self.categories = sorted(list(self.positive_reviews_train.keys()))
                self.category_sizes_train_pos = {category : len(self.positive_reviews_train[category]) for category in self.categories}
                self.category_sizes_train_neg = {category : len(self.negative_reviews_train[category]) for category in self.categories}
                self.indexed_dataset_train = []
                for category in self.positive_reviews_train:
                    for review in self.positive_reviews_train[category]:
                        self.indexed_dataset_train.append([review, category, 1])
                for category in self.negative_reviews_train:
                    for review in self.negative_reviews_train[category]:
                        self.indexed_dataset_train.append([review, category, 0])
                random.shuffle(self.indexed_dataset_train)
            elif train_or_test is 'test':
                if sys.version_info[0] == 3:
                    self.positive_reviews_test, self.negative_reviews_test, self.vocab = pickle.loads(dataset, encoding='latin1')
                else:
                    self.positive_reviews_test, self.negative_reviews_test, self.vocab = pickle.loads(dataset)
                self.vocab = sorted(self.vocab)
                self.categories = sorted(list(self.positive_reviews_test.keys()))
                self.category_sizes_test_pos = {category : len(self.positive_reviews_test[category]) for category in self.categories}
                self.category_sizes_test_neg = {category : len(self.negative_reviews_test[category]) for category in self.categories}
                self.indexed_dataset_test = []
                for category in self.positive_reviews_test:
                    for review in self.positive_reviews_test[category]:
                        self.indexed_dataset_test.append([review, category, 1])
                for category in self.negative_reviews_test:
                    for review in self.negative_reviews_test[category]:
                        self.indexed_dataset_test.append([review, category, 0])
                random.shuffle(self.indexed_dataset_test)

        def get_vocab_size(self):
            return len(self.vocab)

        def one_hotvec_for_word(self, word):
            word_index =  self.vocab.index(word)
            #print(word_index)
            hotvec = torch.zeros(1, len(self.vocab))
            hotvec[0, word_index] = word_index 
            '''Here comes the naive, yet powerful way of improving the accuracy from 0-100 to something decent.
            I do something called integer-hot encoding. As on observation that, just passing an integer array with all indices as 
            suggested in the manual didn't yield better results, and was complex in modifying the arrays. So, this is an easy step using 
            which, though the length is still the same, I integer-hot it, which yields better accuracy (Maybe crossentropy works only for
            1-0 s)'''
            #print(hotvec)
            return hotvec

        def review_to_tensor(self, review):
            review_tensor = torch.zeros(len(review), len(self.vocab))
            for i,word in enumerate(review):
                review_tensor[i,:] = self.one_hotvec_for_word(word)
            return review_tensor

        def sentiment_to_tensor(self, sentiment):
            sentiment_tensor = torch.zeros(2)
            if sentiment is 1:
                sentiment_tensor[1] = 1
            elif sentiment is 0: 
                sentiment_tensor[0] = 1
            sentiment_tensor = sentiment_tensor.type(torch.long)
            return sentiment_tensor

        def __len__(self):
            if self.train_or_test is 'train':
                return len(self.indexed_dataset_train)
            elif self.train_or_test is 'test':
                return len(self.indexed_dataset_test)

        def __getitem__(self, idx):
            sample = self.indexed_dataset_train[idx] if self.train_or_test is 'train' else self.indexed_dataset_test[idx]
            review = sample[0]
            review_category = sample[1]
            review_sentiment = sample[2]
            review_sentiment = self.sentiment_to_tensor(review_sentiment)
            review_tensor = self.review_to_tensor(review)
            category_index = self.categories.index(review_category)
            sample = {'review'       : review_tensor, 
                      'category'     : category_index, # should be converted to tensor, but not yet used
                      'sentiment'    : review_sentiment }
            return sample



    def load_SentimentAnalysisDataset(self, dataserver_train, dataserver_test ):   
        self.train_dataloader = torch.utils.data.DataLoader(dataserver_train,
                    batch_size=1,shuffle=True, num_workers=4)
        self.test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                            batch_size=1,shuffle=False, num_workers=4)
    
    #To avoid issues, I am using an altogether different dataloader function exclusively for my task 3 (different batchsize)
    def load_SentimentAnalysisDatasetTask3(self, dataserver_train, dataserver_test ):   
        self.train_dataloader = torch.utils.data.DataLoader(dataserver_train,
                    batch_size=4,shuffle=True, num_workers=4, collate_fn = pad_collate)
        self.test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                            batch_size=1,shuffle=False, num_workers=4, collate_fn = pad_collate)

    class TEXTnet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(TextClassification.TEXTnet, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.combined_to_hidden = nn.Linear(input_size + hidden_size, hidden_size)
            self.combined_to_middle = nn.Linear(input_size + hidden_size, 100)
            self.middle_to_out = nn.Linear(100, output_size)     
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout(p=0.1)

        def forward(self, input, hidden):
            combined = torch.cat((input, hidden), 1)
            hidden = self.combined_to_hidden(combined)
            out = self.combined_to_middle(combined)
            out = torch.nn.functional.relu(out)
            out = self.dropout(out)
            out = self.middle_to_out(out)
            out = self.logsoftmax(out)
            return out,hidden         

    class TEXTnetOrder2(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(TextClassification.TEXTnetOrder2, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.combined_to_hidden = nn.Linear(input_size + 2*hidden_size, hidden_size)
            self.combined_to_middle = nn.Linear(input_size + 2*hidden_size, 100)
            self.middle_to_out = nn.Linear(100, output_size)     
            self.softmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout(p=0.1)
            # for the cell
            self.cell = torch.zeros(1, hidden_size)
            self.cell = self.cell.to(device)
            self.linear_for_cell_input = nn.Linear(input_size, hidden_size) 
            self.linear_for_cell_hidden = nn.Linear(hidden_size, hidden_size) #Idea courtesy: Prof. Kak. 

        def forward(self, input, hidden):
            hidden_clone = hidden.clone()
            combined = torch.cat((input, hidden, self.cell), 1)
            hidden = self.combined_to_hidden(combined)
            out = self.combined_to_middle(combined)
            out = torch.nn.functional.relu(out)
            out = self.dropout(out)
            out = self.middle_to_out(out)
            out = self.softmax(out)
            input_clone = input.clone()
            lc = self.linear_for_cell_input(input_clone) + self.linear_for_cell_hidden(hidden_clone) #linear combination
            self.cell = torch.sigmoid(lc).detach()
            return out,hidden         

    class GRUnet(nn.Module):
        """
        Source: https://blog.floydhub.com/gru-with-pytorch/ - Courtesy: Prof. Kak 
        """
        def __init__(self, input_size, hidden_size, output_size, n_layers, drop_prob=0.2):
            super(TextClassification.GRUnet, self).__init__()
            self.hidden_size = hidden_size
            self.n_layers = n_layers
            self.gru = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, dropout=drop_prob)
            self.fc = nn.Linear(hidden_size, output_size)
            self.relu = nn.ReLU()
            self.logsoftmax = nn.LogSoftmax(dim=1)
            
        def forward(self, x, h):
            out, h = self.gru(x, h)
            out = self.fc(self.relu(out[:,-1]))
            out = self.logsoftmax(out)
            return out, h

        def init_hidden(self, batch_size):
            weight = next(self.parameters()).data
            hidden = weight.new(self.n_layers, batch_size, self.hidden_size).zero_()
            return hidden

    def save_model(self, model):
        "Save the trained model to a disk file"
        torch.save(model.state_dict(), path_saved_model)


    def run_code_for_training_for_text_classification_with_gru(self, net, hidden_size): 
        filename_for_out = "performance_numbers_" + str(epochs) + ".txt"
        FILE = open(filename_for_out, 'w')
        net = copy.deepcopy(net)
        net = net.to(device)
        ##  Note that the GREnet now produces the LogSoftmax output:
        criterion = nn.NLLLoss()
#            criterion = nn.MSELoss()
#            criterion = nn.CrossEntropyLoss()
        accum_times = []
        optimizer = optim.Adam(net.parameters(), 
                      lr=learning_rate) #Adam has better convergence than SGD. We can ignore momentum parameter as well in adam
        print("+Task1:\n")
        for epoch in range(epochs):  
            running_loss = 0.0
            start_time = time.clock()
            for i, data in enumerate(self.train_dataloader):    
                review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                review_tensor = review_tensor.to(device)
                sentiment = sentiment.to(device)
                optimizer.zero_grad()
                hidden = net.init_hidden(1).to(device)
                for k in range(review_tensor.shape[1]):
                    output, hidden = net(torch.unsqueeze(torch.unsqueeze(review_tensor[0,k],0),0), hidden)
                ## NLLLoss, CrossEntropyLoss
                loss = criterion(output, torch.argmax(sentiment, 1))
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                if i % 100 == 99:    
                    avg_loss = running_loss / float(100)
                    current_time = time.clock()
                    time_elapsed = current_time-start_time
                    print("[epoch:%d  iter:%4d]     loss: %.3f" % (epoch+1,i+1,avg_loss))
                    accum_times.append(current_time-start_time)
                    FILE.write("%.3f\n" % avg_loss)
                    FILE.flush()
                    running_loss = 0.0
        self.save_model(net)

    def run_code_for_testing_text_classification_with_gru(self, net, hidden_size):
        net.load_state_dict(torch.load(path_saved_model))
        classification_accuracy = 0.0
        negative_total = 0
        positive_total = 0
        confusion_matrix = torch.zeros(2,2)
        with torch.no_grad():
            for i, data in enumerate(self.test_dataloader):
                review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                hidden = net.init_hidden(1)
                for k in range(review_tensor.shape[1]):
                    output, hidden = net(torch.unsqueeze(torch.unsqueeze(review_tensor[0,k],0),0), hidden)
                predicted_idx = torch.argmax(output).item()
                gt_idx = torch.argmax(sentiment).item()
                if predicted_idx == gt_idx:
                    classification_accuracy += 1
                if gt_idx is 0: 
                    negative_total += 1
                elif gt_idx is 1:
                    positive_total += 1
                confusion_matrix[gt_idx,predicted_idx] += 1
        out_percent = np.zeros((2,2), dtype='float')
        out_str = "                      "
        out_str +=  "%18s    %18s" % ('predicted negative', 'predicted positive')
        print(out_str + "\n")
        for i,label in enumerate(['true negative', 'true positive']):
            out_percent[0,0] = "%.3f" % (100 * confusion_matrix[0,0] / float(negative_total))
            out_percent[0,1] = "%.3f" % (100 * confusion_matrix[0,1] / float(negative_total))
            out_percent[1,0] = "%.3f" % (100 * confusion_matrix[1,0] / float(positive_total))
            out_percent[1,1] = "%.3f" % (100 * confusion_matrix[1,1] / float(positive_total))
            out_str = "%12s:  " % label
            for j in range(2):
                out_str +=  "%18s" % out_percent[i,j]
            print(out_str)

    def run_code_for_training_for_text_classification_no_gru(self, net, hidden_size):        
        filename_for_out = "performance_numbers_" + str(epochs) + ".txt"
        FILE = open(filename_for_out, 'w')
        net = copy.deepcopy(net)
        net = net.to(device)
        criterion = nn.NLLLoss()
        accum_times = []
        optimizer = optim.Adam(net.parameters(), 
                      lr=learning_rate)
        start_time = time.clock()
        print("\n+Task2:\n")
        for epoch in range(epochs):  
            running_loss = 0.0
            for i, data in enumerate(self.train_dataloader):    
                hidden = torch.zeros(1, hidden_size)
                hidden = hidden.to(device)
                review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                review_tensor = review_tensor.to(device)
                sentiment = sentiment.to(device)
                optimizer.zero_grad()
                input = torch.zeros(1,review_tensor.shape[2])
                input = input.to(device)
                for k in range(review_tensor.shape[1]):
                    input[0,:] = review_tensor[0,k]
                    output, hidden = net(input, hidden)
                loss = criterion(output, torch.argmax(sentiment,1))
                running_loss += loss.item()
                loss.backward(retain_graph=True)        
                optimizer.step()
                if i % 100 == 99:    
                    avg_loss = running_loss / float(100)
                    current_time = time.clock()
                    time_elapsed = current_time-start_time
                    print("[epoch:%d  iter:%4d]     loss: %.3f" % (epoch+1,i+1,avg_loss))
                    accum_times.append(current_time-start_time)
                    FILE.write("%.3f\n" % avg_loss)
                    FILE.flush()
                    running_loss = 0.0
        self.save_model(net)

    def run_code_for_testing_text_classification_no_gru(self, net, hidden_size):
        net.load_state_dict(torch.load(path_saved_model))
        net.to(device)
        classification_accuracy = 0.0
        negative_total = 0
        positive_total = 0
        
        confusion_matrix = torch.zeros(2,2)
        with torch.no_grad():
            for i, data in enumerate(self.test_dataloader):
                hidden = torch.zeros(1, hidden_size)
                hidden = hidden.to(device)
                review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                review_tensor = review_tensor.to(device)
                sentiment = sentiment.to(device)
                input = torch.zeros(1,review_tensor.shape[2])
                input = input.to(device)

                for k in range(review_tensor.shape[1]):
                    input[0,:] = review_tensor[0,k]
                    output, hidden = net(input, hidden)
                predicted_idx = torch.argmax(output).item()
                gt_idx = torch.argmax(sentiment).item()
                if predicted_idx == gt_idx:
                    classification_accuracy += 1
                if gt_idx is 0: 
                    negative_total += 1
                elif gt_idx is 1:
                    positive_total += 1
                confusion_matrix[gt_idx,predicted_idx] += 1
        out_percent = np.zeros((2,2), dtype='float')
        out_str = "                      "
        out_str +=  "%18s    %18s" % ('predicted negative', 'predicted positive')
        print(out_str + "\n")
        for i,label in enumerate(['true negative', 'true positive']):
            out_percent[0,0] = "%.3f" % (100 * confusion_matrix[0,0] / float(negative_total))
            out_percent[0,1] = "%.3f" % (100 * confusion_matrix[0,1] / float(negative_total))
            out_percent[1,0] = "%.3f" % (100 * confusion_matrix[1,0] / float(positive_total))
            out_percent[1,1] = "%.3f" % (100 * confusion_matrix[1,1] / float(positive_total))
            out_str = "%12s:  " % label
            for j in range(2):
                out_str +=  "%18s" % out_percent[i,j]
            print(out_str)

    '''I was facing issues with loading the 4 batchsize data onto the train function. Hence, I used a list, and iterated through 
    the loop to assign values from dictionary to my lists, appended them, and passed them onto GPU device. I tried putting net.init_hidden(4)
    but it didn't seem to work. Then, I realized that since I'm breaking down the batch into single elements, it's sufficient to pass (1)
    as argument.'''

    def run_code_for_training_for_text_classification_with_gru_task3(self, net, hidden_size): 
        filename_for_out = "performance_numbers_" + str(epochs) + ".txt"
        FILE = open(filename_for_out, 'w')
        net = copy.deepcopy(net)
        net = net.to(device)
        ##  Note that the GREnet now produces the LogSoftmax output:
        criterion = nn.NLLLoss()
        accum_times = []
        optimizer = optim.Adam(net.parameters(), 
                      lr=learning_rate)
        print("\n+Task3:\n")
        for epoch in range(epochs):  
            running_loss = 0.0
            start_time = time.clock()
            for i, data in enumerate(self.train_dataloader): 
                review_tensor =[]
                sentiment=[]
                category=[]
                for x in data:
                    review_tensor.append(torch.unsqueeze(x['review'],0))
                    sentiment.append(x['sentiment'])
                    category.append(x['category'])
                optimizer.zero_grad()
                hidden = net.init_hidden(1).to(device)
                output_stack=[]
                for x in range(len(review_tensor)):
                    rev = review_tensor[x].to(device)
                    for k in range(rev.shape[1]):
                        output, hidden = net(torch.unsqueeze(torch.unsqueeze(rev[0,k],0),0), hidden)
                    output_stack.append(output)

                output_stack = torch.squeeze(torch.stack(output_stack)).to(device)
                sentiment = torch.stack(sentiment).to(device)
                ## If using NLLLoss, CrossEntropyLoss
                loss = criterion(output_stack, torch.argmax(sentiment, 1))
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                if i % 100 == 99:    
                    avg_loss = running_loss / float(100)
                    current_time = time.clock()
                    time_elapsed = current_time-start_time
                    print("[epoch:%d  iter:%4d]     loss: %.3f" % (epoch+1,i+1,avg_loss))
                    accum_times.append(current_time-start_time)
                    FILE.write("%.3f\n" % avg_loss)
                    FILE.flush()
                    running_loss = 0.0
        self.save_model(net)

    def run_code_for_testing_text_classification_with_gru_task3(self, net, hidden_size):
        net.load_state_dict(torch.load(path_saved_model))
        net.to(device)
        classification_accuracy = 0.0
        negative_total = 0
        positive_total = 0
        confusion_matrix = torch.zeros(2,2)
        with torch.no_grad():
            for i, data in enumerate(self.test_dataloader):
                review_tensor =[]
                sentiment=[]
                category=[]
                for x in data:
                    review_tensor.append(torch.unsqueeze(x['review'],0))
                    sentiment.append(x['sentiment'])
                    category.append(x['category'])
                hidden = net.init_hidden(1).to(device)
                out_stack=[]
                for x in range(len(review_tensor)):
                    rev = review_tensor[x].to(device)
                    for k in range(rev.shape[1]):
                        output, hidden = net(torch.unsqueeze(torch.unsqueeze(rev[0,k],0),0), hidden)
                    out_stack.append(output)

                out_stack = torch.squeeze(torch.stack(out_stack)).to(device)
                sentiment = torch.stack(sentiment).to(device)

                predicted_idx = torch.argmax(out_stack).item()
                gt_idx = torch.argmax(sentiment).item()
                if predicted_idx == gt_idx:
                    classification_accuracy += 1
                if gt_idx is 0:
                    negative_total += 1
                elif gt_idx is 1:
                    positive_total += 1
                confusion_matrix[gt_idx,predicted_idx] += 1
        out_percent = np.zeros((2,2), dtype='float')
        out_str = "                      "
        out_str +=  "%18s    %18s" % ('predicted negative', 'predicted positive')
        print(out_str + "\n")
        for i,label in enumerate(['true negative', 'true positive']):
            out_percent[0,0] = "%.3f" % (100 * confusion_matrix[0,0] / float(negative_total))
            out_percent[0,1] = "%.3f" % (100 * confusion_matrix[0,1] / float(negative_total))
            out_percent[1,0] = "%.3f" % (100 * confusion_matrix[1,0] / float(positive_total))
            out_percent[1,1] = "%.3f" % (100 * confusion_matrix[1,1] / float(positive_total))
            out_str = "%12s:  " % label
            for j in range(2):
                out_str +=  "%18s" % out_percent[i,j]
            print(out_str)

text_cl = TextClassification() #Task 1
text_c2 = TextClassification() #Task 2
text_c3 = TextClassification() #Task 3

#Task 1
dataserver_train1 = TextClassification.SentimentAnalysisDataset(
                                 train_or_test = 'train',
#                                dataset_file = "sentiment_dataset_train_3.tar.gz",
                                 dataset_file = "sentiment_dataset_train_40.tar.gz",
#                                dataset_file = "sentiment_dataset_train_200.tar.gz",
                                                                      )
dataserver_test1 = TextClassification.SentimentAnalysisDataset(
                                 train_or_test = 'test',
#                                dataset_file = "sentiment_dataset_test_3.tar.gz",
                                 dataset_file = "sentiment_dataset_test_40.tar.gz",
#                                 dataset_file = "sentiment_dataset_test_40.tar.gz",
                                                                  )
#Task 2
dataserver_train2 = TextClassification.SentimentAnalysisDataset(
                                 train_or_test = 'train',
#                                dataset_file = "sentiment_dataset_train_3.tar.gz",
                                 dataset_file = "sentiment_dataset_train_200.tar.gz",
#                                dataset_file = "sentiment_dataset_train_200.tar.gz",
                                                                      )
dataserver_test2 = TextClassification.SentimentAnalysisDataset(
                                 train_or_test = 'test',
#                                dataset_file = "sentiment_dataset_test_3.tar.gz",
                                 dataset_file = "sentiment_dataset_test_200.tar.gz",
#                                 dataset_file = "sentiment_dataset_test_40.tar.gz",
                                                                  )
#Task 3
dataserver_train3 = TextClassification.SentimentAnalysisDataset(
                                 train_or_test = 'train',
#                                dataset_file = "sentiment_dataset_train_3.tar.gz",
                                 dataset_file = "sentiment_dataset_train_40.tar.gz",
#                                dataset_file = "sentiment_dataset_train_200.tar.gz",
                                                                      )
dataserver_test3 = TextClassification.SentimentAnalysisDataset(
                                 train_or_test = 'test',
#                                dataset_file = "sentiment_dataset_test_3.tar.gz",
                                 dataset_file = "sentiment_dataset_test_40.tar.gz",
#                                 dataset_file = "sentiment_dataset_test_40.tar.gz",
                                                                  )
#Task 1
text_cl.dataserver_train = dataserver_train1
text_cl.dataserver_test = dataserver_test1

#Task 2 
text_c2.dataserver_train = dataserver_train2
text_c2.dataserver_test = dataserver_test2

#Task 3
text_c3.dataserver_train = dataserver_train3
text_c3.dataserver_test = dataserver_test3

text_cl.load_SentimentAnalysisDataset(dataserver_train1, dataserver_test1) #Task 1 
text_c2.load_SentimentAnalysisDataset(dataserver_train2, dataserver_test2) #Task 2 
text_c3.load_SentimentAnalysisDatasetTask3(dataserver_train3, dataserver_test3) #Task 3

vocab_size1 = dataserver_train1.get_vocab_size() #Task 1 
vocab_size2 = dataserver_train2.get_vocab_size() #Task 2 
vocab_size3 = dataserver_train3.get_vocab_size() #Task 3

#Common parameters
hidden_size = 512
output_size = 2                            # for positive and negative sentiments
n_layers = 2

#Models initialized for each task 
#model = text_cl.TEXTnet(vocab_size, hidden_size, output_size) - not using anywhere as I wanted to experiment on GRUnet
model1 = text_cl.GRUnet(vocab_size1, hidden_size, output_size, n_layers) #Task 1
model2 = text_cl.TEXTnetOrder2(vocab_size2, hidden_size, output_size) #Task 2
model3 = text_c2.GRUnet(vocab_size3, hidden_size, output_size, n_layers) #Task 3

#Task 1 train and test
text_cl.run_code_for_training_for_text_classification_with_gru(model1, hidden_size)
text_cl.run_code_for_testing_text_classification_with_gru(model1, hidden_size)

#Task 2 train and test - dataset-200 is used here only to demonstrate as this is much faster than GRUnet
text_c2.run_code_for_training_for_text_classification_no_gru(model2, hidden_size)
text_c2.run_code_for_testing_text_classification_no_gru(model2, hidden_size)

#Task 3 train and test
text_c3.run_code_for_training_for_text_classification_with_gru_task3(model3, hidden_size)
text_c3.run_code_for_testing_text_classification_with_gru_task3(model3, hidden_size)

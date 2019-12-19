# Author: Piyush Vyas & Darshan Shinde
# End-to-End Model

import torch
import time
import librosa
import random
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


# Lists all samples (30k samples each gender)
females = librosa.util.find_files('./PS60k/F/', ext=['wav'], recurse=True)
males = librosa.util.find_files('./PS60k/M/', ext=['wav'], recurse=True)


# Randomly Sample 100 utterances per speaker for testing
test_speakers = list()
female_test_speakers = set()
female_set = set(females)

for i in range(30):
    for idx in random.sample(range(i*1000, (i+1)*1000), 100):
        female_test_speakers.add(females[idx])
females = list(female_set - female_test_speakers)    
    

male_test_speakers = set()
male_set = set(males)

for i in range(30):
    for idx in random.sample(range(i*1000, (i+1)*1000), 100):
        male_test_speakers.add(males[idx])
males = list(male_set - male_test_speakers)

test_speakers += female_test_speakers
test_speakers += male_test_speakers


# 54k samples for Training
tr_spks_ = females + males
train_speakers = tr_spks_
train_labels = [i.replace('.wav', '.npy') for i in tr_spks_]
train_speakers.sort(), train_labels.sort()

print('----------------------------------------------------------------')
print('No. of Training Samples =>',len(train_speakers))


# 6k samples for Testing
te_spks_ = test_speakers
test_labels = [i.replace('.wav', '.npy') for i in te_spks_]
test_speakers.sort(), test_labels.sort()

print('----------------------------------------------------------------')
print('No. of Testing Samples =>', len(test_speakers))


# DataLoader Classes for loading Training and Testing data
class trainLoader(data.Dataset):
    def __init__(self):
        self.features, self.labels = gen_features(train_speakers, train_labels, True)
    
    def __getitem__(self, index):
        return torch.tensor(self.features[index], dtype=torch.float32), torch.tensor(self.labels[index], dtype=torch.float32)
        
    def __len__(self):
        return len(self.features)
        

class testLoader(data.Dataset):
    def __init__(self):
        self.features_, self.labels_ = gen_features(test_speakers, test_labels, False)
    
    def __getitem__(self, index):
        return torch.tensor(self.features_[index], dtype=torch.float32), torch.tensor(self.labels_[index], dtype=torch.float32)
        
    def __len__(self):
        return len(self.features_)


# Function to generate features for training and testing the model
def gen_features(spks, labs, Train=True):
    wavs = []
    l = []

    for i in range(len(spks)):
        audio, _ = librosa.load(spks[i], sr=None)
        lab = np.argmax(np.load(labs[i]))
        
        if Train: # Replicate utterance to a fixed length sequence during training
            audio = np.concatenate((audio, audio), axis=None)
            audio = np.concatenate((audio, audio), axis=None)
            audio = np.concatenate((audio, audio), axis=None)
        wavs.append(audio[0:56139])
        l.append(lab)
    return wavs, l


print('-----------------------------------------------------------------')
print('Loading Data.....')
tic=time.time()

print('Loading Train Data!')
trainData = data.DataLoader(trainLoader(), batch_size=128, shuffle=True, drop_last=False)
print('Loading Test Data!')
testData = data.DataLoader(testLoader(), batch_size=1, shuffle=True, drop_last=False)
print('Data Loading Done in ', (time.time()-tic)/60, 'minutes!')


# Model Class
class WavNet(nn.Module):
    def __init__(self):
        super(WavNet, self).__init__()
        
        # 1st convolution layer
        self.conv = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, stride=3, padding=0), nn.BatchNorm1d(num_features=128), nn.LeakyReLU(negative_slope=0.3))
        
        # Residual block 1
        self.res1a = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1), nn.BatchNorm1d(num_features=128), nn.LeakyReLU(negative_slope=0.3), nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1), nn.BatchNorm1d(num_features=128))
        
        self.res1b = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1), nn.BatchNorm1d(num_features=128), nn.LeakyReLU(negative_slope=0.3), nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1), nn.BatchNorm1d(num_features=128))
        
        # Residual block 2
        self.res2a = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1), nn.BatchNorm1d(num_features=256), nn.LeakyReLU(negative_slope=0.3), nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1), nn.BatchNorm1d(num_features=256))
        
        self.res2b = nn.Sequential(nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1), nn.BatchNorm1d(num_features=256), nn.LeakyReLU(negative_slope=0.3), nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1), nn.BatchNorm1d(num_features=256))
        
        self.res2c = nn.Sequential(nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1), nn.BatchNorm1d(num_features=256), nn.LeakyReLU(negative_slope=0.3), nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1), nn.BatchNorm1d(num_features=256))
        
        self.res2d = nn.Sequential(nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1), nn.BatchNorm1d(num_features=256), nn.LeakyReLU(negative_slope=0.3), nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1), nn.BatchNorm1d(num_features=256))
        
        # Recurrent layer with 512 LSTM units
        self.lstm = nn.LSTM(input_size=256, hidden_size=512, batch_first=True)
        
        # Fully connected layer with 128 units
        self.fc1 = nn.Linear(in_features=512, out_features=128)
        
        # Final classification layer
        self.classifier = nn.Linear(in_features=128, out_features=60)
        
        self.leaky = nn.LeakyReLU(negative_slope=0.3)
        self.maxpool = nn.MaxPool1d(kernel_size=3)
        self.downsample1 = nn.Conv1d(128, 128, kernel_size=1)
        self.downsample2 = nn.Conv1d(128, 256, kernel_size=1)
        
        
    def forward(self, x):
        self.lstm.flatten_parameters()
            
        x = self.conv(x)

        out = self.res1a(x)
        out += self.downsample1(x) # identity connection
        out = self.leaky(out)
        x = self.maxpool(out)
            
        out = self.res1b(x)
        out += self.downsample1(x) # identity connection
        out = self.leaky(out)
        x = self.maxpool(out)
            
        out = self.res2a(x)
        out += self.downsample2(x) # identity connection
        out = self.leaky(out)
        x = self.maxpool(out)
            
        out = self.res2b(x)
        out += x # identity connection
        out = self.leaky(out)
        x = self.maxpool(out)
        
        out = self.res2c(x)
        out += x # identity connection
        out = self.leaky(out)
        out = self.maxpool(out)
            
        out = self.res2d(x)
        out += x # identity connection
        out = self.leaky(out)
        x = self.maxpool(out)
        
        x = x.permute(0, 2, 1)   
        x, _ = self.lstm(x)
            
        x = self.fc1(x[:, -1, :])
            
        x = self.classifier(x)
            
        return F.log_softmax(x)


# Instantiate the model class
model = WavNet()
model = nn.DataParallel(model).cuda() # Multi-GPU Training
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.003) # Adam optimizer
criterion = nn.NLLLoss() # Neg-Log Likelihood Loss


# Train Function to train model
def train(model, trainData, testData, optimizer, criterion):
    trainLoss = 0
    correct = 0
    for idx, (data, target) in enumerate(trainData):
        model.train()
        data, target = data.cuda(), target.long().cuda()
        optimizer.zero_grad()
        output = model(data.unsqueeze_(1))
        loss = criterion(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        trainLoss += loss.item()
        loss.backward()
        optimizer.step()
    return trainLoss, 100.*(correct/len(trainData.dataset))


# Test function to test/evaluate model
def test(model, data_, target_):
    model.eval()
    with torch.no_grad():
        data_, target_ = data_.cuda(), target_.long().cuda()
        output_ = model(data_.unsqueeze_(1))
        pred_ = output_.argmax(dim=1)
        if target_ != pred_:
            return 1
        else: 
            return 0

# Driver code
def main(model, trainData, testData, optimizer, criterion, epochs):
    for epoch in range(epochs):
        eval_dict = defaultdict(lambda:(0, 0))
        loss, train_acc = train(model, trainData, testData, optimizer, criterion)
        for i in range(60):
            for idx, (data_, target_) in enumerate(testData):
                FR, IA = eval_dict[target_]
                if i == idx//900:
                    eval_dict[target_] = (FR + test(model, data_, target_), IA) # add 1 if pred != actual in case of sample of current speaker
                else:
                    eval_dict[target_] = (FR, IA + test(model, data_, target_)) # add 1 if pred != actual in case of sample of other speaker
        
        FR, IA = 0, 0
        for FR_, IA_ in eval_dict.values():
            FR += (FR_/6000)/6
            IA += (IA_/6000)/6
        print('Epoch:', epoch, 'Train Batch Loss:', loss, 'Train Acc:', train_acc, 'Imposter Accept:', 100.* IA, 'False Reject:', 100.* FR)


if __name__ == '__main__':
    main(model, trainData, testData, optimizer, criterion, epochs=26)
            
            
        
        




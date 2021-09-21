from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32768, 4096)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x=torch.flatten(x,1)
        #print(x.size())
        x = self.fc1(x)
        #print(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        #print("Output is",output)
        return output

class CLNet(nn.Module):
    def __init__(self):
        super(CLNet, self).__init__()
        #Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        #self.fc1 = nn.Linear(12544, 128)
        #self.fc2 = nn.Linear(128, 2)
        self.L1=nn.LSTM(input_size=1290496,hidden_size=10,batch_first=True)
        #self.L2=nn.LSTM()
        self.fc1 = nn.Linear(40, 2)
        #self.fc2 = nn.Linear(128, 2)
    
    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.conv1(x)
        #print("Size after conv1:",x.size())
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        #print("Size after cov2:",x.size())
        x = F.max_pool2d(x, 2)

        #print("Size after pooling:",x.size())
        x = self.dropout1(x)
        
        #Flattens a contiguous range of dims in a tensor
        #x = torch.flatten(x, 1)
        #print("Size after flattern:", x.size())
        y = x.view(batch_size, timesteps, -1)      
        output,(hn,cn) = self.L1(y)

        #print(output.size())
        output = output.reshape(batch_size,-1)
        #print(output.size(),"after reshape")
        output = self.dropout2(output)
        output = self.fc1(output)
        output = F.relu(output)
        #print(output.size(),"linear")
        output = F.log_softmax(output, dim=1)
        
        return output
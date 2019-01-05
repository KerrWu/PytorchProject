
# coding: utf-8

# In[1]:

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import Dropout, Linear, Conv2d, MaxPool2d, AvgPool2d, ReLU, Sequential, Softmax, BatchNorm2d, CrossEntropyLoss
from torch.optim import SGD
from torch.nn.init import xavier_normal, normal


# In[2]:

class MyNet(nn.Module):
    
    def __init__(self):
        super(MyNet, self).__init__()
        
        self.conv1 = Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = MaxPool2d(kernel_size=2, stride=2)
        self.relu1 = ReLU()
        self.fc1 = Linear(in_features=150*150*4, out_features=6)
        self.dropout1 = Dropout(p=0.5)
        
        self.softmax = Softmax()

        for m in self.modules():
            if isinstance(m, Conv2d):
                xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, Linear):
                normal(m.weight.data, 0, 0.01)
                m.bias.data.zero_()
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)
        x = x.view(-1,150*150*4)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.softmax(x)
        
        return x

    # def initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, Conv2d):
    #             xavier_normal(m.weight.data)
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, Linear):
    #             normal(m.weight.data, 0, 0.01)
    #             m.bias.data.zero_()


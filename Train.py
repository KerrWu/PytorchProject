
# coding: utf-8

# In[1]:

import MyDataset
import MyNet
import torch
import torchvision
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss,DataParallel
from torch.cuda import device_count, is_available
from torch.autograd import Variable


# In[2]:

epoches = 10
batchSize = 8
baseLR = 0.01
useReduceLR = True
useEarlyStop = True
usePretrainedModel = False

# 如果usePretrainedModel = True， 则需使用该路径导入模型参数，该路径应导向一个pkl文件
modelPath = '...'

# 训练数据
trainTxtPath = './data/train.txt'
trainData = MyDataset.MyDataset(txtPath=trainTxtPath, transforms=True, targetTransform='train')
trainLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True, num_workers=2, drop_last=False)
batchNum = len(trainLoader)


# 验证数据
validTxtPath = './data/valid.txt'
validData = MyDataset.MyDataset(txtPath=validTxtPath, transforms=True, targetTransform='valid')
validLoader = DataLoader(validData, batch_size=batchSize, shuffle=True, num_workers=2, drop_last=False)
valNum = validData.__len__()


# In[ ]:

'''
自动判断是否具有多GPU环境，如果有，默认使用所有GPU
'''

net = MyNet.MyNet()
if usePretrainedModel:
    net.load(modelPath)
    
else:
    net.initialize_weights()

if device_count() > 1:
    print("Find {0} devices in this machine".format(device_count()))
    net = DataParallel(net)
    
if is_available():
    print("gpu is available in this machine")
    net.cuda()
else:
    print("can not find any gpu device on this machine, use cpu mode")


# In[ ]:

optimizer = SGD(net.parameters(), lr=baseLR)

#如果使用学习率减少策略，则定义相关的类
if useReduceLR:
    lrScheduler = ReduceLROnPlateau(optimizer,mode="max",factor=0.8, patience=50, verbose=True,cooldown=10,min_lr=baseLR*0.00001)

lossFunc = CrossEntropyLoss()

# 开始训练
for epoch in range(epoches):
    
    lossSum = 0
    batch = 0
    
    # 1个epoch
    for i, data in enumerate(trainLoader):
        
        batch += 1
        
        inputs,labels = data
        
        
        #如果使用gpu，则需要将数据放在主gpu上
        if is_available():
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = lossFunc(outputs, labels)
        loss.backward()
        optimizer.step()
        lossSum += loss.data[0]
        
        print("epoch = {epoch}, batch = {batch}/{batchNum}, loss = {loss}".format(epoch=epoch, batch=batch, batchNum=batchNum, loss=loss))
        
        
    #训练1个epoch后，计算整个valid dataset上的表现,给出val accu，如果使用reduce lr，则由val accu确定是否reduce
    correctPred = 0
    for vi,vdata in enumerate(validLoader):

        vinputs,vlabels = vdata

        if is_available():
            vinputs = Variable(vinputs.cuda())
            vlabels = Variable(vlabels.cuda())        
        else:
            vinputs = Variable(vinputs)
            vlabels = Variable(vlabels)

        voutputs = net(vinputs)
        _, vpred = torch.max(voutputs, 1)

        result = sum(vpred==vlabels).numpy()
        correctPred += result

    valAccu = correctPred/valNum

    if useReduceLR:
        lrScheduler.step(valAccu)
        
    print("val accuracy = {valAccu}".format(valAccu=valAccu))
        
        


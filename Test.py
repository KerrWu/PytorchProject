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

# 测试参数
batchSize = 8
modelPath = "..."
useMultiGPU = False

# 测试数据
testTxtPath = './data/test.txt'
testData = MyDataset.MyDataset(txtPath=testTxtPath, transforms=True, targetTransform='test')
testLoader = DataLoader(testData, batch_size=batchSize, shuffle=True, num_workers=2, drop_last=False)
testNum = testData.__len__()

# 载入模型
net = MyNet.MyNet()
net.load(modelPath)

if device_count() > 1 and useMultiGPU:
    print("Find {0} devices in this machine".format(device_count()))
    net = DataParallel(net)

if is_available():
    print("gpu is available in this machine")
    net.cuda()
else:
    print("can not find any gpu device on this machine, use cpu mode")

correctPred = 0

for i,data in enumerate(testLoader):

    inputs,labels = data

    if is_available():
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
    else:
        inputs = Variable(inputs)
        labels = Variable(labels)

    outputs = net(inputs)
    _, pred = torch.max(outputs, 1)

    result = sum(pred==labels).numpy()
    correctPred += result

valAccu = correctPred/testNum
import MyDataset
import MyNet
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.cuda import is_available
from torch.autograd import Variable

# 测试参数
modelPath = "..."

# 测试数据
testTxtPath = './data/test.txt'
testData = MyDataset.MyDataset(txtPath=testTxtPath, transforms=True, targetTransform='test')
testLoader = DataLoader(testData, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

# 载入模型
net = MyNet.MyNet()
net.load(modelPath)

if is_available():
    print("gpu is available in this machine")
    net.cuda()
else:
    print("can not find any gpu device on this machine, use cpu mode")

for i,data in enumerate(testLoader):

    imgName = testData.imgsAndLabels[i]

    input, label = data

    if is_available():
        input = Variable(input.cuda())
        label = Variable(label.cuda())
    else:
        input = Variable(input)
        label = Variable(label)

    outputs = net(input)
    _, pred = torch.max(outputs, 1)

    print("input:{imgName}, result:{pred}".format(imgName=imgName, pred=pred))

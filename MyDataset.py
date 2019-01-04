
# coding: utf-8

# In[1]:

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import os

#预处理采用标准化或是归一化？NORM为True则为标准化，反之为归一化
NORM = True
img_h, img_w = (300,300)


# In[4]:

class MyDataset(Dataset):
    def __init__(self, txtPath, transforms=None, targetTransform=None):
        '''初始化dataset类，
        args: txt文件路径， 是否进行预处理, 数据预处理方式标志由targetTransform传入，为train或valid或test
        '''
        f = open(txtPath, 'r')
        self.imgsAndLabels = []
        for line in f:
            line = line.strip()
            words = line.split()
            self.imgsAndLabels.append( (words[0], words[1]) )
        f.close()  
        
        
        if NORM:
            self.means, self.stdevs = self.normalize_func()
            
        if transforms is not None:       
            self.transforms = self.transforms_func(targetTransform)
        else:
            self.transforms = None

    def __getitem__(self, index):
        '''每次调用返回1个数据
        args: list的索引
        return: img, label
        '''
        
        img, label = self.imgsAndLabels[index]
        img = Image.open(img).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
            
        return img, int(label)
    
    def __len__(self):
        return len(self.imgsAndLabels)        
    
    
    def transforms_func(self, dataset):
        '''
        数据预处理函数，对训练和验证(测试)使用不同的处理方式
        不加入Normalize是归一化，加入则变为标准化,是否加入由全局变量NORM控制   
        如果需要标准化，则首先需要对全部训练数据计算均值和标准差并记录下来
        由于transforms_func在每次getitem都会被调用，
        因此计算均值和标准差的过程需要在init时就一次做完，
        这样之后每次调用时就不用重复计算
        '''
        assert (dataset in ['train', 'valid', 'test'])

        if NORM:

            if dataset == 'train':
                trainTransforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(self.means,self.stdevs)
                ])
                return trainTransforms

            if dataset == 'test' or dataset == 'valid':
                validTransforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(self.means,self.stdevs)
                ])
                return validTransforms

        else:

            if dataset == 'train':
                trainTransforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ToTensor()
                ])
                return trainTransforms

            if dataset == 'test' or dataset == 'valid':
                validTransforms = transforms.Compose([
                    transforms.ToTensor()
                ])
                return validTransforms
        
    def normalize_func(self):
        
        '''
        mean、std保存在./norm.txt文件中，如果存在，说明之前已经计算过，直接载入即可，否则需要重新计算
        文件内容格式为：
        means r g b
        stdevs r g b       
        '''
        
        if os.path.isfile('./norm.txt'):
            
            with open('./norm.txt','r') as f:
            
                lines = f.readlines()
                
                assert (len(lines) == 2)
                means = lines[0].rstrip().split()[1:]
                stdevs = lines[1].rstrip().split()[1:]
                
                means = [float(elem) for elem in means]
                stdevs = [float(elem) for elem in stdevs]
            
        else: 
        
            CNum = len(self.imgsAndLabels)
            imgs = np.zeros([img_w, img_h, 3, 1])
            means, stdevs = [], []

            for i in range(CNum):
                img_path = self.imgsAndLabels[i][0].rstrip().split()[0]

                img = cv2.imread(img_path)
                img = cv2.resize(img, (img_h, img_w))

                img = img[:, :, :, np.newaxis]
                imgs = np.concatenate((imgs, img), axis=3)
                print(i)

            imgs = imgs.astype(np.float32)/255.


            for i in range(3):
                pixels = imgs[:,:,i,:].ravel()  # 拉成一行
                means.append(np.mean(pixels))
                stdevs.append(np.std(pixels))

            means.reverse() # BGR --> RGB
            stdevs.reverse()
            
            with open('./norm.txt','w') as f:
            
                f.write('means')
                f.write(' ')
                for mean in means:
                    f.write(str(mean)+' ')
                f.write('\n')

                f.write('stdevs')
                f.write(' ')
                for std in stdevs:
                    f.write(str(std)+' ')
                f.write('\n')        

        print("normMean = {}".format(means))
        print("normStd = {}".format(stdevs))
        print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))
        
        return means, stdevs


# In[5]:

# txtPath = './data/train.txt'
# train_data = MyDataset(txtPath, transforms=True, targetTransform='train')
# train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=1, drop_last=False)


# In[6]:

# for i, data in enumerate(train_loader):
#     img, label = data
#     print(img)


# In[ ]:




# In[ ]:




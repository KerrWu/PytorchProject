
# coding: utf-8

# In[1]:

import os
import argparse
import sys

# CLASSNAME = [
# 'basal-cell-carcinoma',
# 'lupus-erythematosus',
# 'rosacea',
# 'seborrheic-keratosis',
# 'solar-keratosis',
# 'squamous-carcinoma']

CLASSNAME = ['car', 'cow', 'human','zebra','airplane']


# In[2]:

'''
输入参数为train/valid/test中的一个，读取data下对应目录的文件并在主目录下生成同名的对应txt文件
'''

parser = argparse.ArgumentParser()


# In[3]:

parser.add_argument('-d', '--dataset', type=str, choices=['train', 'valid', 'test'])


# In[4]:

args = parser.parse_args(sys.argv[1:])
dataPath = os.path.join('./data/', args.dataset)


# In[5]:

classNum = 0
classList = []

for elem in os.listdir(dataPath):
    if os.path.isdir(os.path.join(dataPath, elem)):
        classNum += 1
        classList.append(os.path.join(dataPath, elem))
        
txtPath = os.path.join('./data', args.dataset+'.txt')

with open(txtPath, 'w') as f:
    
    for elem in classList:
        
        for file in os.listdir(elem):
            
            if os.path.splitext(file)[-1] == '.jpg':
                f.write(os.path.join(elem, file))
                f.write(' ')
                f.write(str(CLASSNAME.index(os.path.split(elem)[-1])))
                f.write('\n')


# In[ ]:




# 模版内容：Pytorch实现CV项目

**生成txt.py**

传入参数dataset为 train/valid/test 中的一个，读取./data/dataset/下文件，不同类别的文件放入不同文件夹中，文件夹名即类别名

在./data目录下生成 dataset.txt文件，其内包含了所有数据，其格式为

```
./data/test/car/000000000001.jpg 0
./data/test/car/000000000108.jpg 0
./data/test/zebra/000000000080.jpg 3
./data/test/airplane/000000000463.jpg 4
./data/test/airplane/000000000191.jpg 4
./data/test/human/000000000202.jpg 2
./data/test/human/000000000016.jpg 2
./data/test/human/000000000171.jpg 2
./data/test/human/000000000063.jpg 2
./data/test/human/000000000057.jpg 2
./data/test/human/000000000069.jpg 2
./data/test/human/000000000345.jpg 2
./data/test/cow/000000000090.jpg 1
./data/test/cow/000000000019.jpg 1 
```



**MyDataset.py**

由生成的txt文件创建Dataset类，其中包含了数据预处理

按Pytorch的要求继承Dataset类并重写\__getitem\___和 \_\_len\_\_

transform：是否进行预处理

targetTransfom：数据预处理方式，train/valid/test

NORM：预处理采用标准化还是归一化



**MyNet**

创建网络结构，Pytorch要求继承nn.Module并重写forward

以Resnet作为例子



**Train.py**

训练，如果检测到GPU自动使用，且如果有多块GPU则将模型替换为DataParallel



**Test.py**

测试，同样检测到GPU自动使用
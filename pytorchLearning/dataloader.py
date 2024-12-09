"""
DataLoader 的使用说明

dataset=要处理的数据集
batch_size = 一次要处理的数量
shuffle = 是否按照顺序取还是打乱
num_workers = 不重要
drop_last =  当数量不足batch_size的时候，是否舍去
"""

import torchvision 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#加载数据集
test_data = torchvision.datasets.CIFAR10("./CIFAR10" , train = False,transform=torchvision.transforms.ToTensor(),download=True)

#使用DataLoader
test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)

#测试数据集中的第一张图片及target
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("logs")
for epoch in range(2): #遍历两轮
    step = 0
    for data in test_loader:
        imgs,targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch:{}".format(epoch),imgs,step)#注意这里是add_images()
        step+=1

writer.close()

        

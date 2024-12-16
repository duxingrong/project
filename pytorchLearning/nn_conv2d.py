"""
卷积层的例子
"""
import torch 
import torchvision
from  torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10("data",train = False , transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset=dataset , batch_size= 64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    
    def forward(self,input):
        output = self.conv1(input)
        return output
    

tudui  = Tudui()
writer = SummaryWriter("logs")

step = 0
for data in dataloader:
    imgs,targets = data
    print(imgs.shape)
    writer.add_images("imgs",imgs, step)
    output= tudui(imgs)
    #由于彩色图像只有三通道才能显现，6变成3
    output = torch.reshape(output ,(-1,3,30,30))
    writer.add_images("output" , output , step)
    print(output.shape)
    step+=1 


writer.close()  #别忘记了

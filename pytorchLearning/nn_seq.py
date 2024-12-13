"""
搭建以下CIFAR的模型
"""

import torch 
import torchvision
from torch import nn
from torch.nn import Conv2d , MaxPool2d, Linear , Sequential,Flatten
from torch.utils.tensorboard import SummaryWriter 


class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = Sequential(
            Conv2d(3,32,5,padding=2), #这里根据公式来算出Padding 和stride
            MaxPool2d(2),
            Conv2d(32,32,5 , padding = 2),
            MaxPool2d(2),
            Conv2d(32, 64 ,5,  padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )

    def forward(self,x):
        output = self.model1(x)
        return output 
    


tudui  =Tudui()
input = torch.ones((64,3,32,32))
output = tudui(input)
print(output.shape)

#模型依旧可以在tensorboard中展开
writer= SummaryWriter("logs")
writer.add_graph(tudui , input )
writer.close()
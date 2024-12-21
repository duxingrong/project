"""
创建针对CIFAR10的神经网络
"""

import torch 
import torchvision
from torch import nn
from torch.nn import Conv2d , MaxPool2d, Linear , Sequential,Flatten



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
    
if __name__=="__main__":
    inputs = torch.ones((64,3,32,32))
    tudui = Tudui()
    outputs = tudui(inputs)
    print(outputs.shape)
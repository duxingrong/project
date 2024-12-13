"""
线性层，就是将输入的长度修改成指定的长度
比如在数字识别中，我们只需要0-9,那么长度指定为10(输出为10)
"""
import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Linear

dataset = torchvision.datasets.CIFAR10("data" , train=False , transform=torchvision.transforms.ToTensor() , download=True)

dataloader = DataLoader(dataset , batch_size=64,drop_last=True)# 舍去，不然后面线形层输入不对

class TuDui(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(196608 , 10)
        
    def forward(self,input ):
        output  = self.linear(input)
        return output
    
tudui = TuDui()

for data in dataloader:
    imgs , targets = data
    output = torch.flatten(imgs) #摊平，成一维
    print(output.shape)
    output = tudui(output)
    print(output.shape)
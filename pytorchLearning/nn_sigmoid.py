"""
介绍非线性激活函数，作用就是引入的越多，模型的拟合效果就更好
"""
import torch 
import torchvision 
from torch import nn 
from torch.utils.tensorboard import SummaryWriter 
from torch.utils.data import  DataLoader
from torch.nn import ReLU , Sigmoid

#首先测试以下ReLU 
input  = torch.tensor([[1,-0.5],
                       [-1, 2 ]])

input = torch.reshape(input ,(-1,1, 2,2))
print(input.shape)

relu = ReLU( inplace=False) #不在原地修改值
print(relu(input))

dataset = torchvision.datasets.CIFAR10("data" , train  = False , transform=torchvision.transforms.ToTensor() , download=True )

dataloader = DataLoader(dataset , batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = Sigmoid()

    def forward(self,input):
        output = self.sigmoid(input)
        return output
    

tudui = Tudui()
writer = SummaryWriter("logs")
step = 0

for data in dataloader:
    imgs, targets  = data
    writer.add_images("imgs",imgs,step)
    output  = tudui(imgs)
    writer.add_images("output",output,step)
    step +=1 

writer.close()

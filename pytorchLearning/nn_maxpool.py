"""
最大池化层，就是一种有损压缩，减少数据量
"""
import torch 
import torchvision
from torch import  nn
from torch.nn import MaxPool2d

from torch.utils.tensorboard import SummaryWriter 
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("data" , train = False , transform=torchvision.transforms.ToTensor() , download=  True  )

dataloader = DataLoader(dataset , batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = MaxPool2d(kernel_size=3 , ceil_mode=True)# ceil_mode True就是不足也保留 ，False 就是舍弃

    def forward(self, input):
        output = self.maxpool(input)

        return output 
    
#实例化
tudui = Tudui()
writer= SummaryWriter("logs")

step = 0
for data in dataloader:
    imgs , targets = data
    writer.add_images("imgs" , imgs, step)
    output  = tudui(imgs)
    writer.add_images("maxpool" , output ,step)
    step+=1 

writer.close()
    

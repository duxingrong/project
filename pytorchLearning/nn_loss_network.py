"""
神经网络中如何使用损失函数
"""
import torch 
import torchvision
from torch import nn
from torch.nn import Conv2d , MaxPool2d, Linear , Sequential,Flatten 
from torch.utils.data import DataLoader

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
    dataset = torchvision.datasets.CIFAR10('data',train = False,transform=torchvision.transforms.ToTensor(),download=True)
    dataloader= DataLoader(dataset,batch_size=1)
    tudui = Tudui()
    loss = nn.CrossEntropyLoss()
    for data in dataloader:
        imgs,targets = data
        outputs = tudui(imgs)
        #计算实际输出和目标之间的差距
        #为我们更新输出提供一定的依据(反向传播)
        result_loss = loss(outputs,targets)
        result_loss.backward() #反向传播
        print(result_loss)


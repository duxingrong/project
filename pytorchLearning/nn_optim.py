"""
优化器的使用
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
    


if __name__ == "__main__":
    # 检查是否有 GPU，如果有就用 GPU，没有就使用 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 下载数据集
    dataset = torchvision.datasets.CIFAR10('data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
    dataloader = DataLoader(dataset, batch_size=64)  # 使用较大的 batch_size
    tudui = Tudui().to(device)  # 将模型迁移到设备上
    
    # 损失函数
    loss = nn.CrossEntropyLoss()
    
    # 创建优化器
    optim = torch.optim.SGD(tudui.parameters(), lr=0.01)
    
    # 训练循环
    for epoch in range(20):
        running_loss = 0.0
        for data in dataloader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)  # 将数据迁移到设备上
            
            # 前向传播
            outputs = tudui(imgs)
            
            # 计算损失
            result_loss = loss(outputs, targets)
            
            # 反向传播
            optim.zero_grad()  # 清零梯度
            result_loss.backward()  # 反向传播
            optim.step()  # 更新参数
            
            running_loss += result_loss.item()  # 累计损失值
            
        print(f"Epoch [{epoch+1}/20], Loss: {running_loss:.4f}")
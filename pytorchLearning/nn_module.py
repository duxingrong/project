"""
简单介绍了骨架nn.Module
"""

import torch 
from torch import nn

class Tudui(nn.Module):
    
    def __init__(self):
        super().__init__() #继承必写的


    def forward(self,input):
        output = input +1 
        return output
    


if __name__=="__main__":
    #实例化
    tudui = Tudui()
    x = torch.tensor(1.0)
    ouput = tudui(x) #__call__方法中调用了forward函数
    print(ouput)
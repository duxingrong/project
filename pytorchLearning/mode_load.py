"""
模型的加载
"""
import torch 
import torchvision


#方式1->保存方式1,加载模型
model = torch.load("vgg16_method1.pth")
# print(model)


#方式2,加载模型
vgg16 = torchvision.models.vgg16(weights=None)
vgg16.load_state_dict(torch.load('vgg16_method2.pth'))#传入保存的模型参数
# model = torch.load("vgg16_method2.pth")
print(vgg16)


#陷阱,第一种保存方式的时候,需要我们引入类,不然系统不知道
from model_save import Tudui 

tudui = torch.load("tudui_method1.pth")
print(tudui)
"""
如何使用一些现有的模型,并且进行一些修改
"""
import torchvision
from torchvision.models import VGG16_Weights
from torch import nn
#不加载与训练权重和加载预训练权重
vgg16_false = torchvision.models.vgg16(weights=None)
vgg16_true = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

print(vgg16_true)

#在现有的模型上添加一层
vgg16_true.classifier.add_module("add_linear" , nn.Linear(1000,10))
print(vgg16_true)



#在原来的模型上修改
print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096,10)
print(vgg16_false)
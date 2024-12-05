"""
tensorboard的学习使用,目前的理解就是一个画图的工具
"""
from torch.utils.tensorboard import SummaryWriter 
import numpy as np
from PIL import Image 
writer  = SummaryWriter("logs")

image_path = r"C:\Users\Diamond\Desktop\project\pytorchLearning\dataset\train\ants\5650366_e22b7e1065.jpg"
img_PIL = Image.open(image_path)
#由于writer的函数只接受tensor类型或者numpy类型，所以引入numpy
img_array = np.array(img_PIL)
print(type(img_array))


writer.add_image("test" , img_array,1,dataformats="HWC") #是numpy就需要修改这个参数


for i in range(100):
    writer.add_scalar("y=x" , i , i )

writer.close()

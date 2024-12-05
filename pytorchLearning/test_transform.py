"""
主要是介绍transform 如何使用，他就是一个工具箱，可以对图片进行很多处理，常用的方法是将PIL或者np.array转成tensor张量
"""

from torchvision import transforms
from PIL import Image 
from torch.utils.tensorboard import SummaryWriter
# 1. 如何使用transforms
img_path = r"C:\Users\19390\Desktop\project\pytorchLearning\dataset\train\ants\0013035.jpg"
img = Image.open(img_path)



#实例化
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(type(tensor_img))#可以发现将类型转成了tensor 

writer = SummaryWriter("logs")
writer.add_image("Tensor_img",tensor_img,1)

writer.close()

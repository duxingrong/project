"""
Dataset的实战训练
"""
from torch.utils.data import Dataset
from PIL import Image 
import matplotlib.pyplot as plt 
import os

class MyData(Dataset):

    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)  #将所有的图片储存进了列表

    def __getitem__(self,idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.path,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)
    
root_dir ="C:\\Users\\Diamond\\Desktop\\project\\pytorchLearning\\dataset\\train"
ant_label_dir = "ants"
bees_label_dir = "bees" 

#实例化
ant_dataset = MyData(root_dir , ant_label_dir)
bees_dataset = MyData(root_dir , bees_label_dir)

ant_len = ant_dataset.__len__()
bees_len = bees_dataset.__len__()
print(ant_len)
print(bees_len)

train_dataset = ant_dataset+bees_dataset 
print(train_dataset.__len__())

img, label = train_dataset.__getitem__(125)
plt.imshow(img)
plt.axis("off")
plt.show()

"""
不过一般的情况label都是在train 目录里有两个文件，一个是ants_image ,一个是ants_label,然后里面的文件名保持一致，一个记录数据的图片，一个记录数据的label ，这里由于名字就是label，所以简化了
"""



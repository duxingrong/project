import cv2
import numpy as np
import random


img = cv2.imread('colorcolor.jpg')

#查看图片的格式
print(img.shape)
print(type(img))

#opencv中是bgr顺序,图片的(0,0)到左上角，x右正，y下正


#创建自己的图片
#img = np.empty((300,300,3),np.uint8)

#修改图片
#for  row in range(300):
#    for col in range(img.shape[1]):
#        img[row][col] = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]



#切割图片
new_img = img[400:650,100:500] #第一值是高度，第二个值是宽度
cv2.imshow('img',img)
cv2.imshow('new_img',new_img)
cv2.waitKey(0)
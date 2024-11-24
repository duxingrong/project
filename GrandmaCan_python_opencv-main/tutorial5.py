import cv2
import numpy as np

img =np.zeros((600,600,3),np.uint8)

#画直线
cv2.line(img, (0,0),(img.shape[1],img.shape[0]),(0,255,0),2)

#画方形
cv2.rectangle(img, (0,0),(400,300),(0,0,255),2)
#填满就cv2.FILLED换粗度

#画圆形
cv2.circle(img, (300,400),100,(255,0,0),2)

#写文字,位置是左下角的坐标，不支持中文
cv2.putText(img, 'hello',(100,500),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),2)









cv2.imshow('img',img)
cv2.waitKey(0)
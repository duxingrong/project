import numpy as np
import cv2
#常用函数

img = cv2.imread('colorcolor.jpg')
img = cv2.resize(img,(0,0),fx =0.5, fy = 0.5)
#cv2.imshow('img',img)


#bgr转成灰阶
#gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray',gray )
#cv2.waitKey(0)

#高斯模糊,第二个是核大小，只能是奇数,第三个标准差
#blur = cv2.GaussianBlur(img, (15,15),10)
#cv2.imshow('blur',blur)
#cv2.waitKey(0)


#图像边缘检测,最低门槛和最高门槛
canny = cv2.Canny(img, 200, 300)
cv2.imshow('canny',canny)
#cv2.waitKey(0)

#膨胀,第二个核，第三个次数
kernel = np.ones((5,5),np.uint8)
dilate = cv2.dilate(canny,kernel, iterations=1)
cv2.imshow('dilate',dilate )
#cv2.waitKey(0)

#侵蚀，第二个核，第三个次数
kernel2 = np.ones((5,5),np.uint8)
erode = cv2.erode(dilate,kernel2, iterations=1)
cv2.imshow('erode',erode )
cv2.waitKey(0)
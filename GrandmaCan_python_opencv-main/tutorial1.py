import cv2

#读取图片
img = cv2.imread('colorcolor.jpg')


#resize两种方法
#img = cv2.resize(img, (400,500))
img = cv2.resize(img, (0,0), fx = 0.5, fy = 0.5)

#显示图片
cv2.imshow('img',img)
cv2.waitKey(0)

import cv2

img = cv2.imread('shape.jpg')

imgContour = img.copy()

#轮廓检测
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(img, 150,200)
contours,hierarchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

for cnt in contours:
    cv2.drawContours(imgContour,cnt,-1, (255,0,0),4)
    area = cv2.contourArea(cnt)#面积
    if area >500:
        #print(cv2.arcLength(cnt, True))#边长
        peri = cv2.arcLength(cnt,True)
        vertices =cv2.approxPolyDP(cnt, peri*0.02,True)
        corners= len(vertices)
        x,y ,w, h = cv2.boundingRect(vertices)
        cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),4)
        if corners  ==3 :
            cv2.putText(imgContour,'trangle',(x,y-5),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        elif corners == 4:
            cv2.putText(imgContour,'rectrangle',(x,y-5),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        elif corners==5:
            cv2.putText(imgContour,'emtagen',(x,y-5),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        elif corners>=6:
            cv2.putText(imgContour,'circle',(x,y-5),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
   


cv2.imshow('canny',canny)
cv2.imshow('img',img)
cv2.imshow('imgContour',imgContour)
cv2.waitKey(0)
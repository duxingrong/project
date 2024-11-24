import cv2
import numpy as np



penColorHSV = [[42,143,0,119,195,231]]

penColorBGR = [[255,0,0],[0,255,0],[0,255,255]]


drawPoints = []



cap = cv2.VideoCapture(0)
#找到笔
def findPen(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for i in range(len(penColorHSV)):

        lower = np.array(penColorHSV[i][:3])
        upper = np.array(penColorHSV[i][3:])
        mask = cv2.inRange(hsv,lower,upper)
        result = cv2.bitwise_and(img, img, mask=mask,)
        pen_x, pen_y = findContour(mask)
        cv2.circle(imgContour,(pen_x,pen_y),20,penColorBGR[i],cv2.FILLED)
        if pen_x!= -1:
            drawPoints.append([pen_x,pen_y,penColorBGR[i]])
        #cv2.imshow('result',result)


#设置笔尖
def findContour(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    x,y,w,h =-1,-1,-1,-1
    for cnt in contours:
        #cv2.drawContours(imgContour,cnt,-1, (255,0,0),4)
        area = cv2.contourArea(cnt)#面积
        if area >500:
            #print(cv2.arcLength(cnt, True))#边长
            peri = cv2.arcLength(cnt,True)
            vertices =cv2.approxPolyDP(cnt, peri*0.02,True)
            x,y ,w, h = cv2.boundingRect(vertices)

    return x+w//2, y


#画轨迹
def draw(drawpoints):
    for point in drawpoints:
        cv2.circle(imgContour,(point[0],point[1]),20,point[2],cv2.FILLED)



while True:
    ret, frame  =cap.read()
    if ret :
        imgContour = frame.copy()
        #frame = cv2.resize(frame,(0,0),fx=0.5,fy =0.5)
        cv2.imshow('frame',frame)
        findPen(frame)
        draw(drawPoints)
        cv2.imshow('contour',imgContour)
    else:
        break
    if cv2.waitKey(1)==ord('q'):
        break
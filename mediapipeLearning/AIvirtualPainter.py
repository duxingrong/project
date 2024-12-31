"""
利用opencv和mediapipe来制作画画的工具
"""

import cv2 
import os 
import HandTrackingModule as htm
import numpy as np

# 首先读取图片
folderPath = "mediapipeLearning/Header"
imgList = os.listdir(folderPath)

overLaylist = [] #存放图片
for imPath in imgList:
    image = cv2.imread(f"{folderPath}/{imPath}")
    overLaylist.append(image)

# 实例化
handdetector = htm.HandDetector(detectionCon=0.7)


# 变量
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
imgCanvas = np.zeros((720,1280,3),np.uint8)


colorR = (255,0,255)
colorB = (225,0,0)
colorG = (0,255,0)
colorBlack = (0,0,0)
color = (255,0,255)

radiusW = 20
radiusE = 50
radius = 5
HeaderIndex = 0

## prev time point
px,py  = 0,0 


if __name__=="__main__":
    
    # 主程式
    while True:
        success , img  = cap.read()
        if success:
            
            # 镜像 , 这样我们才能想往右边，就往右边
            img = cv2.flip(img,1)
            # 检测人手
            Hands,img = handdetector.findHands(img,flipType=False)
            if Hands:
                lmList = Hands[0]['lmList']
                ## 标出食指和中指
                p1 = lmList[8][:2]
                p2 = lmList[12][:2]
                
                fingers = handdetector.fingersUp(Hands[0],flipType=False)
                print(fingers)
                ## 根据位置来改变图片
                if fingers[1] ==1 and fingers[2]==1:
                    # print('Select Mode')
                    px,py = 0,0
                    if p1[1]<125:
                        if 0<p1[0]<350:
                            ## 改变颜色 以及半径
                            color = colorR
                            radius = radiusW
                            HeaderIndex = 0 
                        elif 350<p1[0]<600:
                            color= colorG
                            radius = radiusW
                            HeaderIndex = 1 
                        elif 600<p1[0]<800:
                            color = colorB
                            radius = radiusW
                            HeaderIndex = 2 
                        elif 800<p1[0]<1280:
                            color = colorBlack
                            radius = radiusE
                            HeaderIndex = 3 
                
                ## Drawing
                if fingers[1]==1 and fingers[2]==0:
                    # print('Draw Mode')
                    if px!=0 and py != 0:
                        ## 画画
                        cv2.line(img ,(px,py),p1,color,radius)
                        cv2.line(imgCanvas ,(px,py),p1,color,radius)
                        # print(px,py,p1)
                    px,py = p1


                ## 如果所有手指举起，就把imgCanvas重置，全黑
                if fingers == [1,1,1,1,1]:
                    imgCanvas = np.zeros((720,1280,3),np.uint8)


                ## 这里之所以实现了画画的功能，是通过imgCanvas上画画,然后提取其中的非0像素,然后与img进行融合实现的
                imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY) # 先转为灰度图
                _, imgInv =  cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV) #二值化，背景变成全白，而线条变成全黑
                imgInv = cv2.cvtColor(imgInv , cv2.COLOR_GRAY2BGR)  # 转回BGR
                img  = cv2.bitwise_and(img,imgInv)  # 这一步保证了在img中呈现线条，但是是黑色,因为前面二值化
                img = cv2.bitwise_or(img , imgCanvas) #再把上面显示的黑色Line和imgCanvas或操作，这样黑色的线条变成彩色，并且由于imgCanvas的背景是黑色，或运算后不影响img的背景

                """
                bitwise_and 有一个0像素,就是0
                bitwise_or 有一个非0,就是非0
                """

                cv2.circle(img ,p1,radius,color,cv2.FILLED)
            img[:125,:] = overLaylist[HeaderIndex]

            cv2.imshow("Image",img)
            key = cv2.waitKey(1)
            if key==ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    
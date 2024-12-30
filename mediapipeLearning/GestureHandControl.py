"""
使用mediapipe 和 htm模块,来实现手势数字识别
"""

import cv2 
import HandTrackingModule as htm 
import time 
import os
# 实例化
handdetector = htm.HandDetector(detectionCon=0.75)

## 获取手势图片路径
folderPath="mediapipeLearning/picture"
piclist = os.listdir(folderPath)
# print(piclist)

overLayList = []
for imPath in piclist:
    image = cv2.imread(f"{folderPath}/{imPath}")
    overLayList.append(image)

# print(len(overLayList))


if __name__=="__main__":
    


    # 设置摄像机
    cap = cv2.VideoCapture(0)

    # 变量
    cap.set(3,1280)
    cap.set(4,720)
    pTime = 0
    # 主循环
    while True:
        success , img = cap.read()
        if success :
            
            ## 开启手追踪
            allHands,img = handdetector.findHands(img)

            if allHands:
                ## 只针对一只手
                Hand = allHands[0]
                fingers = handdetector.fingersUp(Hand)
                print(fingers.count(1))

                ## 引入对应的手势图片
                h,w,c = overLayList[fingers.count(1)].shape
                img[:h,:w] = overLayList[fingers.count(1)-1] # 这里减一是因为从0开始

                ## 添加画图的美观
                cv2.rectangle(img ,(0,200),(100,300),(255,255,255),cv2.FILLED)
                cv2.putText(img ,str(fingers.count(1)),(30,270),cv2.FONT_HERSHEY_PLAIN,5,(0,0,0),5)
            

            ## FPS
            cTime = time.time()
            FPS = int(1/(cTime-pTime))
            pTime = cTime
            cv2.putText(img ,f"FPS:{FPS}",(1100,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)



            cv2.imshow('Image',img )
            key = cv2.waitKey(1)
            if key==ord('q'):
                break
    

    cap.release()
    cv2.destroyAllWindows()
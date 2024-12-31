"""
使用身体检测,计算角度来设计的一款训练打卡脚本
"""
import cv2 
import time 
import PoseModule as pm
import numpy as np 
# 实例化
posedetecetor = pm.PoseDetector(detectionCon=0.7)

# 变量
pTime = 0
angleBar = 500
anglePer = 0 
count = 0
tmp =0
colorB = (255,0,0)
if __name__=="__main__":
    
    ## 设置摄像机
    cap = cv2.VideoCapture("motionCapture/crossover.mp4")
    
    while True:
        success,img = cap.read()
        if success:
            
            img = cv2.resize(img, (1280,720))

            ## 身体检测
            allPoses ,img = posedetecetor.findPose(img,draw=False)
            if allPoses:
                pose = allPoses[0]
                lmList = pose['lmList']
                
                ## 计算角度
                p1,p2,p3 = lmList[11][:2],lmList[13][:2],lmList[15][:2]
                angle,img = posedetecetor.findAngle(p1,p2,p3,img)
                """
                angle :[140,290]
                anglePer: [0,100]
                angleBar :[500,150]
                """
                anglePer = np.interp(angle,(140,290),(0,100))
                angleBar = np.interp(angle,(140,290),(500,150))
            
                ## 计数器
                if posedetecetor.angleCheck(angle,160,5):
                    if tmp == 0:
                        tmp=1
                        colorB = (255,0,255)
                        count+=0.5
                if posedetecetor.angleCheck(angle,210,5):
                    if tmp == 1:
                        tmp = 0
                        colorB = (255,0,0)
                        count+=0.5

            ## 计数器
            cv2.rectangle(img,(0,520),(200,720),(0,0,0),cv2.FILLED)
            cv2.putText(img,str(int(count)),(50,670),cv2.FONT_HERSHEY_PLAIN,10,(255,255,255),10)
            
                    

            ## 柱状图和百分比
            cv2.rectangle(img,(1100,150),(1200,500),colorB,3)
            cv2.rectangle(img, (1100,int(angleBar)),(1200,500),colorB,cv2.FILLED)
            cv2.putText(img , f"{int(anglePer)}%",(1100,550),cv2.FONT_HERSHEY_PLAIN,3,colorB,3)
            ## FPS
            cTime = time.time()
            FPS = int(1/(cTime-pTime))
            pTime = cTime
            cv2.putText(img , f"FPS:{FPS}",(1100,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)




            cv2.imshow("Image",img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


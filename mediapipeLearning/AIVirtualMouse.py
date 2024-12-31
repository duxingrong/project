"""
结合opencv和autopy来制作一个虚拟鼠标
"""

import cv2 
import numpy as np
import HandTrackingModule as htm 
import time 
import autopy

# 变量
wCam ,hCam = 1280,640
pTime = 0
smoothening = 7
pre_x_mouse = 0
pre_y_mouse = 0


# 实例化
handdetector = htm.HandDetector(maxHands=1,detectionCon=0.75)



if __name__=="__main__":
    
    # 设置相机
    cap = cv2.VideoCapture(0)
    cap.set(3,wCam)
    cap.set(4,hCam)

    # 查看电脑的鼠标移动范围  1920-1080 
    wScr, hScr = autopy.screen.size()
    print(f"w,h={wScr, hScr}")
    


    # 主程式 
    while True:
        success , img = cap.read()
        if success :
            img = cv2.flip(img,1)
            # 手追踪
            Hands,img = handdetector.findHands(img,flipType=False)
            if Hands:
                Hand = Hands[0]
                lmList = Hand['lmList']
                x1,y1 = lmList[8][:2] #食指
                x2,y2 = lmList[12][:2] #中指

                fingers= handdetector.fingersUp(Hand,flipType=False)

            
                ## 映射，鼠标的范围和我框的范围
                x_mouse = np.interp(x1,(200,1080),(0,1920))
                y_mouse = np.interp(y1,(100,620),(0,1080))

                ## 添加润滑，让鼠标移动更顺滑
                if pre_x_mouse!=0  and pre_y_mouse!=0:
                    x_mouse = pre_x_mouse + (x_mouse-pre_x_mouse)/smoothening
                    y_mouse = pre_y_mouse + (y_mouse-pre_y_mouse)/smoothening
                    autopy.mouse.move(x_mouse, y_mouse)
                pre_x_mouse = x_mouse
                pre_y_mouse = y_mouse

                ## 如果食指和中指的距离小于阈值，就算鼠标点击
                length,info  = handdetector.findDistance((x1,y1),(x2,y2))
                cx,cy = info[4],info[5]
                if length <40:
                  cv2.circle(img,(cx,cy),7,(0,255,0),cv2.FILLED)
                  autopy.mouse.click()
                


            cv2.rectangle(img , (200,100),(1080,620),(255,255,255),5)
            # FPS
            cTime = time.time()
            FPS = int(1/(cTime-pTime))
            pTime = cTime
            cv2.putText(img,f"FPS:{FPS}",(40,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

            #show Image
            cv2.imshow("Image",img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break



    cap.release()
    cv2.destroyAllWindows()
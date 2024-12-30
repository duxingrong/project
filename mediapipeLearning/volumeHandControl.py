"""
利用opencv结合mediapipe来控制电脑的音量大小
需要引入包mediapipe opencv-python pycaw 
"""

import cv2 
import HandTrackingModule as htm 
import time
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np





if __name__=="__main__":

    #设置摄像头
    cap = cv2.VideoCapture(0)


    # 实例化
    ## htm
    handdetecotr = htm.HandDetector(detectionCon=0.7)
    ## pycaw 
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)

    # print(volume.GetVolumeRange()) 这一步查看电脑的音量范围 -65.25-0.0
    # volume.SetMasterVolumeLevel(-20.0, None)   这个设置电脑的音量



    ## 变量
    cap.set(3,1280)
    cap.set(4,720)
    pTime = 0
    volumeBar = 400 
    volumePer = 0

    # 主程式
    while True:
        success , img = cap.read()
        if success:
            ## FPS
            cTime =time.time()
            FPS = int(1/(cTime-pTime))
            pTime = cTime
            cv2.putText(img ,f'FPS:{FPS}',(40,70),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),2)
            ## HandTracking 
            allHands ,img = handdetecotr.findHands(img)
            if allHands:
                ## 一只手就够了
                Hand = allHands[0]
                lmList = Hand['lmList']
                ## 计算拇指和食指的距离
                p1 = lmList[4][:2]
                p2 = lmList[8][:2]
                length ,info , img = handdetecotr.findDistance(p1,p2,img)
                ## 修正length长度
                if length<20:  length =20
                if length>280: length =280

                """
                利用Numpy来映射
                length : [20,280]
                volume : [-65.25,0,0]
                柱形的高度: [400,150]
                百分比   : [0,100]
                """
                volumeVal = np.interp(length , [20,280],[-65,0])
                volumeBar = np.interp(length , [20,280],[400,150])
                volumePer = np.interp(length , [20,280],[0,100])
                volume.SetMasterVolumeLevel(volumeVal, None)

            ## 画图
            cv2.rectangle(img , (40,150),(150,400),(255,0,0),2)
            cv2.rectangle(img , (40,int(volumeBar)),(150,400),(255,0,0),cv2.FILLED)
            cv2.putText(img,f'{str(int(volumePer))}%',(40,450),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),3)

            

            cv2.imshow("Image",img)
            key = cv2.waitKey(1)
            if key==ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()



    
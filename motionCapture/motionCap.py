"""
利用opencv 和mediapipe完成动捕
"""

import cv2 
from cvzone.PoseModule import PoseDetector


# 获取视频
cap = cv2.VideoCapture(r"C:\Users\19390\Desktop\project\motionCapture\Video.mp4")

detector = PoseDetector()
posList = []

# 确保视频文件打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


while True :
    success , img = cap.read()
    # 动捕
    img = detector.findPose(img)
    lmList , bboxInfo = detector.findPosition(img)

    if bboxInfo:
        lmString =''
        for lm in lmList:
          lmString += f'{lm[0]},{img.shape[0] - lm[1]},{lm[2]},'
        posList.append(lmString)
    print(len(posList))

        
    cv2.imshow("motioncap" , img)
    
    #提取出数据记录成文本
    key = cv2.waitKey(1)
    if key == ord('q'):
        with open("AnimationFile.txt" ,'w') as f:
            f.writelines(["%s\n" % item for item in posList])

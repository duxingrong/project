"""
利用opencv 和mediapipe完成动捕
"""

import cv2 
from cvzone.PoseModule import PoseDetector


# 获取视频
cap = cv2.VideoCapture(r"C:\Users\19390\Desktop\project\motionCapture\crossover.mp4")

detector = PoseDetector()
posList = []
paused = False

# 确保视频文件打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


while True :
    if not paused:
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

    key = cv2.waitKey(10)
    # 处理按键事件
    if key == ord('q'):  # 按 'q' 键退出
        # with open("AnimationFile.txt" ,'w') as f:
        #     f.writelines(["%s\n" % item for item in posList])
        break
    elif key == ord('s'):  # 按 'j' 键暂停
        paused = True
    elif key == ord('g'):  # 按 'k' 键继续
        paused = False

    cv2.imshow("motioncap", img)


# 释放视频对象并关闭窗口
cap.release()
cv2.destroyAllWindows()
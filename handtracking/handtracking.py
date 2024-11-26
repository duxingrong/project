"""
安装opencv-python 和 mediapipe
"""
import cv2
import mediapipe as mp 
import time 
import os

if not os.path.exists(r"C:\Users\19390\Desktop\project\GrandmaCan_python_opencv-main\face_detect.xml"):
    print("Error: face_detect.xml not found!")
else:
    print("face_detect.xml loaded successfully.")

faceCascade = cv2.CascadeClassifier(r"C:\Users\19390\Desktop\project\GrandmaCan_python_opencv-main\face_detect.xml")


#读取镜头
cap  = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(False , 2, 1, 0.9 ,0.5) #五个参数， 第一个是否是静态的画面(False)，第二个最多几只手(2), 第三个模型复杂度(1)，越大越精确 ， 第四个严谨度(0.5) ,第五个追踪严谨度(0.5)  
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color = (0,0,255) , thickness = 5)
handConStyle = mpDraw.DrawingSpec(color = (0,255,0) , thickness = 10)

pTime = 0
cTime = 0


def detect_gesture(hand_landmarks):
    """
    根据手的关键点判断手势
    """
    #手指的状态列表[大拇指 ， 食指，中指，无名指，小指] ， 1表示竖起，0表示弯曲
    fingers = []

    #提取关键点
    lm = hand_landmarks.landmark
    #判断大拇指是否弯曲
    fingers.append(1  if lm[4].x<lm[3].x else 0)
    #判断其他手指尖
    fingers += [1  if lm[tip].y <lm[tip-1].y else 0  for tip in [8,12,16,20]]    

    #返回竖起的数量来判断手势
    return sum(fingers)



while True:
    ret, img = cap.read()
    if ret:
        #检测人脸 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faceRect = faceCascade.detectMultiScale(gray, 1.1 , 5)
        print(faceRect)#看找到了几个人脸，会返回每个人脸的x,y,w,h      
        for x,y,w,h in faceRect:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

        #检测手
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)
        #print(result.multi_hand_landmarks)# 打印出侦测到手的21个坐标
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        #总的手势值
        total_gesture = 0
        #将所有手以及手的点和线画出来
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                #mpDraw.draw_landmarks(img, handLms,mpHands.HAND_CONNECTIONS,handLmsStyle,handConStyle)
                #得到所有手的点坐标找出来
                for i , lm in enumerate(handLms.landmark):
                    xPos = int(lm.x * imgWidth)#比例*图片大小 = 实际坐标
                    yPos = int(lm.y * imgHeight)
                    #cv2.putText(img , str(i), (xPos-25 , yPos+5), cv2.FONT_HERSHEY_COMPLEX ,0.4,(0,0,255),2)#坐标点序号可视化
                    #将大拇指标红
                    #if i ==4: 
                        #cv2.circle(img , (xPos,yPos),20,(166,56,56),cv2.FILLED)
                    print(i, xPos,yPos)
                #判断手势猜数字
                total_gesture += detect_gesture(handLms)
            cv2.putText(img , f"gesture:{total_gesture}",(imgWidth-300,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)
                    

        #求fps
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img , f"FPS:{int(fps)}", (30,50) , cv2.FONT_HERSHEY_COMPLEX , 1, (255,0,0) ,3)

        cv2.imshow('img',img)
    else:
        break
    if cv2.waitKey(1) == ord('q'):
        break 



import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np 

# Variables
width ,height = 1280 , 720
folderPath = r"C:\Users\19390\Desktop\project\HandGestureControlPresentation\Presentation"


# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)

# Get the list of presetation Images
pathImages =sorted(os.listdir(folderPath),key = len) # 按数字排序
# print(pathImages)

# Variables
imageNumber = 0
hs,ws = int(120*1),int(213*1)
buttonPressed = False
buttonCounter = 0
buttonDelay = 15
annotations = [[]]
annotationNumber = -1
annotationStart = False



# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)
gestureThreshold = 300

while True:
    # Import Images
    success , img  = cap.read()
    img = cv2.flip(img, 1)
    # 导入presentation
    pathFullImage = os.path.join(folderPath,pathImages[imageNumber])
    imgCurrent = cv2.imread(pathFullImage)
    imgCurrent = cv2.resize(imgCurrent,(1280,720))

    hands , img = detector.findHands(img )
    cv2.line(img , (0,gestureThreshold) ,(width,gestureThreshold) ,(0,255,0),10)

    if hands and not buttonPressed: 
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx,cy = hand['center']
        lmList = hand['lmList']


        # Constrain values for easier drawing 
        # indexFinger = lmList[8][0] , lmList[8][1]
        xVal = int(np.interp(lmList[8][0],[width//2,width],[0,width]))
        yVal = int(np.interp(lmList[8][1],[150,height-150],[0,height]))
        indexFinger  = xVal ,yVal


        print(fingers)

        if cy<=gestureThreshold: # if hand is at the height at the face 

            # Gesture 1 - Left
            if fingers == [1,0,0,0,0]:
                # print('Left') 
                annotationStart = False
                
                if imageNumber>0 :
                    buttonPressed =True
                    # update  
                    annotations = [[]]
                    annotationNumber = -1
                    
                    imageNumber -=1

            # Gesture 2 - Right
            if fingers == [0,0,0,0,1]:
                annotationStart = False
                # print('Right') 
                if imageNumber<len(pathImages)-1 :
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = -1
                    
                    imageNumber +=1 
        



        # Gesture 3 - Show Pointer
        if fingers == [0,1,1,0,0]:
            cv2.circle(imgCurrent  , indexFinger , 12 , (0,0,255),cv2.FILLED)
    
        # Gesture 4 - Draw Pointer
        if fingers == [0,1,0,0,0]:
            if annotationStart == False:
                annotationStart = True
                annotationNumber +=1 
                annotations.append([])
            cv2.circle(imgCurrent  , indexFinger , 12 , (0,0,255),cv2.FILLED)
            annotations[annotationNumber].append(indexFinger)
        else:
            annotationStart = False


        # Gesture 5 - Erase 
        if fingers == [0,1,1,1,0]:
            if annotations:
                annotations.pop(-1)
                annotationNumber -=1
                print(annotationNumber)
                buttonPressed =  True        

    else :
        annotationStart = False 



    # Button Pressed itterations
    if buttonPressed == True:
        buttonCounter+=1 
        if buttonCounter>buttonDelay:
            buttonCounter = 0
            buttonPressed = False


    for i in range(len(annotations)):
        for j in range(len(annotations[i])):
            if j>0:
                cv2.line(imgCurrent ,annotations[i][j-1] , annotations[i][j] , (0,0,200),12)



    # Adding wecams  image on the slides
    imgSmall = cv2.resize(img , (ws,hs))
    h,w,c = imgCurrent.shape
    imgCurrent[0:hs,w-ws:w] = imgSmall


    cv2.imshow("Slides",imgCurrent)
    # cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

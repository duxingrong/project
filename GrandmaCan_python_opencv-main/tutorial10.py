import cv2

faceCascade = cv2.CascadeClassifier(r"C:\Users\19390\Desktop\project\GrandmaCan_python_opencv-main\face_detect.xml")



cap = cv2.VideoCapture(0)

while True:
    ret , frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faceRect = faceCascade.detectMultiScale(gray, 1.1 , 5)
        print(faceRect)#看找到了几个人脸，会返回每个人脸的x,y,w,h

        for x,y,w,h in faceRect:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            
        cv2.imshow('frame',frame)
    else:
        break
    if cv2.waitKey(1) == ord('q'):
        break

import cv2

#读取影像

cap = cv2.VideoCapture('thumb.mp4')

#获取摄像机镜头
#cap = cv2.VideoCapture(0)




while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (0,0), fx = 0.5, fy = 0.5)
        cv2.imshow('img',frame)
    else:
        break
    if cv2.waitKey(10)== ord('q'):#结束
        break


"""
Hand Landmarks Detection with MediaPipe Tasks
"""
import cv2 
import mediapipe as mp 
import time 
import math

class HandDetector():
    def __init__(self,mode = False,maxHands=2,detectionCon = 0.5,trackCon = 0.5):
        self.mode = mode 
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(#是否是静态的画面,最多几只手(2),严谨度(0.5),追踪严谨度(0.5) 
            static_image_mode=self.mode,
            max_num_hands = self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence =self.trackCon,
             )    
        self.mpDraw = mp.solutions.drawing_utils\
        
        #指尖序号
        self.tipId = [4,8,12,16,20]

    # 获取手的位姿和画图
    def findHands(self,img,draw=True,flipType=True):
        # 首先传入的opencv的图像是bgr，转成RGB
        imgRGB= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB) 
        h,w,c = img.shape #获取图像的高度和宽度
        allHands = []
        if self.results.multi_hand_landmarks:
            #使用zip 同时遍历两个列表,每次迭代得到一堆数据,一个是Handedness,一个是手的关键点
            for handType ,handlms in zip(self.results.multi_handedness,self.results.multi_hand_landmarks):
                myHand = {}

                ## lmList
                mylmList = []
                x_list = []
                y_list = []
                for id, lm in enumerate(handlms.landmark):
                    px,py,pz = int(lm.x*w),int(lm.y*h),int(lm.z*w)
                    mylmList.append([px,py,pz])
                    x_list.append(px)
                    y_list.append(py)
                
                ## bbox 
                x_min,x_max = min(x_list),max(x_list)
                y_min ,y_max =min(y_list) ,max(y_list)
                bboxW,bboxH = x_max-x_min , y_max-y_min 
                bbox = (x_min,y_min,bboxW,bboxH)
                cx,cy = bbox[0]+(bbox[2])//2 , bbox[1]+(bbox[3])//2
                
                #赋值
                myHand['lmList'] = mylmList
                myHand['bbox']   = bbox 
                myHand['center'] = (cx,cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right" 
                else:
                    myHand["type"] = handType.classification[0].label 

                allHands.append(myHand)            

                if draw: 
                    self.mpDraw.draw_landmarks(img,handlms,self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img,(bbox[0]-20,bbox[1]-20),(bbox[0]+bbox[2]+20,bbox[1]+bbox[3]+20)
                                  ,(255,0,255),2)
                    cv2.putText(img, myHand["type"],(bbox[0]-30,bbox[1]-30),cv2.FONT_HERSHEY_PLAIN,
                                2,(255,0,255),2)
                    
        return allHands,img 
    
    # 判断手的竖起个数，将myHand 作为传入参数
    def fingersUp(self,myHand,flipType=True):
        fingers = []  #结果集
        myHandType = myHand['type']
        mylmList  = myHand['lmList']
        # 在检测到手的前提下
        if self.results.multi_hand_landmarks:
            if flipType==True:
                #大拇指
                if myHandType=="Right":
                    if mylmList[self.tipId[0]][0] > mylmList[self.tipId[0]-1][0]: #4.x>3.x
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:
                    if mylmList[self.tipId[0]][0] < mylmList[self.tipId[0]-1][0]: #4.x<3.x
                        fingers.append(1)
                    else:
                        fingers.append(0)
            else:
                #大拇指
                if myHandType=="Right":
                    if mylmList[self.tipId[0]][0] > mylmList[self.tipId[0]-1][0]: #4.x>3.x
                        fingers.append(0)
                    else:
                        fingers.append(1)
                else:
                    if mylmList[self.tipId[0]][0] < mylmList[self.tipId[0]-1][0]: #4.x<3.x
                        fingers.append(0)
                    else:
                        fingers.append(1)



            # 其他四根手指
            for id in range(1,5):
                if mylmList[self.tipId[id]][1] > mylmList[self.tipId[id]-2][1]:
                    fingers.append(0)
                else:
                    fingers.append(1)

        return fingers 
        

    def findDistance(self,p1,p2,img=None,color=(255,0,255),scale = 5): #scale是半径
        """
        传入的p1.p2是元组,代表点在图片上的(x,y)
        返回的是距离,info,图片
        """
        x1,y1 = p1
        x2,y2 = p2 
        cx,cy = (x1+x2)//2,(y1+y2)//2
        info = (x1,y1,x2,y2,cx,cy)
        length = math.hypot(x2-x1,y2-y1)
        if img is not None:
            cv2.circle(img,(x1,y1),scale,color,cv2.FILLED)
            cv2.circle(img,(x2,y2),scale,color,cv2.FILLED)
            cv2.line(img,(x1,y1),(x2,y2),color,max(1,scale//3))
            cv2.circle(img,(cx,cy),scale,color,cv2.FILLED)
        
        return length , info , img
            

## test 
def main():

    # 实例化
    handdetector = HandDetector()
    # 获取图像
    cap = cv2.VideoCapture(0)


    while True:
        success,img = cap.read()
        #获取手的位姿和图像
        Hands,img = handdetector.findHands(img,draw=True, flipType=True)

        if Hands:
            Hand1 = Hands[0] #主要操作一只手
            lmList1 = Hand1['lmList']
            Hand1Type = Hand1['type']
            Hand1bbox = Hand1['bbox']
            Hand1center = Hand1['center']

            #检测手指弯曲函数
            fingers = handdetector.fingersUp(Hand1)
            print(f"H!={fingers.count(1)}",end=" ")

            #检测距离函数
            length , info , img = handdetector.findDistance(lmList1[8][:2],lmList1[12][:2],img = img )

            if len(Hands) == 2:
                Hand2 = Hands[1]
                lmList2 = Hand2['lmList']
                Hand2Type = Hand2['type']
                Hand2bbox = Hand2['bbox']
                Hand2center = Hand2['center']

                fingers2 = handdetector.fingersUp(Hand2)
                print(f"H2={fingers2.count(1)}",end=" ")

                length,info,img = handdetector.findDistance(lmList2[8][:2],lmList2[12][:2],img=img)

            print(" ")

        cv2.imshow('img',img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    
    #释放
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


        

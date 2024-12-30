"""
身体位资追踪
"""
import math 
import cv2 
import mediapipe  as mp 


class PoseDetector():
    def __init__(self,staticMode=False,modelComplexity=1,smoothLandmarks=True,
                 enableSegmentation=False,smoothSegmentation=True,detectionCon=0.5,
                 trackCon=0.5):
        # 变量
        self.staticMode = staticMode
        self.modelComplexity = modelComplexity
        self.smoothLandmarks = smoothLandmarks
        self.enableSegmentation = enableSegmentation
        self.smoothSegmentation = smoothSegmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        #实例化
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.staticMode,
                                     model_complexity=self.modelComplexity,
                                     smooth_landmarks=self.smoothLandmarks,
                                     enable_segmentation=self.enableSegmentation,
                                     smooth_segmentation=self.smoothSegmentation,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon
                                
                                    ) #多人检测
    def findPose(self,img,draw=True,bboxWithHands=False):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h,w,c  = img.shape
        # 变量
        allPoses = []

        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:

            # 这里没有办法迭代，只能检测画面中的一个人
            PoseLms  = self.results.pose_landmarks

            myPose = {}

            ## lmList
            lmList = []
            for id, lm in enumerate(PoseLms.landmark):
                px,py,pz = int(lm.x*w),int(lm.y*h),int(lm.z*w)
                lmList.append([px,py,pz])

            ##bbox 
            ad = abs(lmList[12][0] - lmList[11][0])//2
            if bboxWithHands:
                x1 = lmList[16][0] - ad 
                x2 = lmList[15][0] + ad 
            else:
                x1 = lmList[12][0] - ad 
                x2 = lmList[11][0] + ad 
            
            y1 = lmList[1][1]  - ad 
            y2 = lmList[29][1] + ad 

            bbox = (x1, y1, x2 - x1, y2 - y1)
            cx ,cy = bbox[0]+(bbox[2])//2,bbox[1]+(bbox[3])//2

            # 赋值
            myPose['lmList'] = lmList
            myPose['bbox']   = bbox 
            myPose['center'] = (cx,cy)

            allPoses.append(myPose)

            if draw:
                self.mpDraw.draw_landmarks(img,PoseLms,self.mpPose.POSE_CONNECTIONS)
                cv2.rectangle(img,bbox,(255,0,255),3)
                cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)

        return allPoses , img 
    
    def findDistance(self,p1,p2,img=None,color=(255,0,255),scale=5):
        x1,y1 = p1
        x2,y2 = p2 
        cx,cy = (x1+x2)//2,(y1+y2)//2
        length  = math.hypot(x2-x1,y2-y1)
        info = (x1,y1,x2,y2,cx,cy)

        if img is not None:
            cv2.line(img,(x1,y1),(x2,y2),color,max(1,scale//3))
            cv2.circle(img,(x1,y1),scale,color,cv2.FILLED)
            cv2.circle(img,(x2,y2),scale,color,cv2.FILLED)
            cv2.circle(img,(cx,cy),scale,color,cv2.FILLED)
        return length ,img , info 

    def findAngle(self,p1,p2,p3,img=None,color=(0,0,255),scale=10):
        x1,y1 = p1
        x2,y2 = p2
        x3,y3 = p3

        angle = math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))

        if angle<0:
            angle +=360
        
        if img is not None :
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), max(1,scale//5))
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), max(1,scale//5))
            cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)
            cv2.circle(img, (x1, y1), scale+5, color, max(1,scale//5))
            cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), scale+5, color, max(1,scale//5))
            cv2.circle(img, (x3, y3), scale, color, cv2.FILLED)
            cv2.circle(img, (x3, y3), scale+5, color, max(1,scale//5))
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, color, max(1,scale//5))

        return angle,img     
    

    def angleCheck(self,angle,targetAngle,offset=20):
        return targetAngle-offset <angle<targetAngle+offset

def main():
    cap  = cv2.VideoCapture(r"C:\Users\19390\Desktop\project\motionCapture\crossover.mp4")

    posedetector = PoseDetector()
    while True:
        success , img = cap.read()
        if success:
            allPoses,img = posedetector.findPose(img,bboxWithHands=True)
            lmList = allPoses[0]['lmList']
            print(len(lmList))
            # print(allPoses)

            length , img , info = posedetector.findDistance(lmList[11][:2],lmList[15][:2],img)
            print(length)

            angle , img = posedetector.findAngle(lmList[11][:2],lmList[13][:2],lmList[15][:2],img=img,scale=10,color=(0,0,255))
            # print(angle)
            
            isAngle140 = posedetector.angleCheck(angle,140,10)
            print(isAngle140)

        cv2.imshow('img',img)

        key= cv2.waitKey(1)
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindow()
    
if __name__=="__main__":
    main()

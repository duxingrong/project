"""
Face Mesh Module
"""
import cv2 
import mediapipe as mp 
import math 

class FaceMeshDetector():

    def __init__(self,staticMode=False,maxFace=2,minDetectionCon=0.5,minTrackCon=0.5):
        """
        :param staticMode: In static mode, detection is done on each image: slower
        :param maxFaces: Maximum number of faces to detect
        :param minDetectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.staticMode = staticMode
        self.maxFace = maxFace
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        
        #实例化mediapipe
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode = self.staticMode,
            max_num_faces  = self.maxFace,
            min_detection_confidence = self.minDetectionCon,
            min_tracking_confidence = self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1,circle_radius=2,color = (0,255,0))


    def findFaceMesh(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = [] #存放所有人脸的数据
        h,w,c = img.shape
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,faceLms,self.mpFaceMesh.FACEMESH_CONTOURS,
                                               self.drawSpec,self.drawSpec)
                
                face = []
                for lm in faceLms.landmark:
                    px,py = int(lm.x*w),int(lm.y*h)
                    face.append([px,py])
                faces.append(face)
            
        return faces,img 
                    

    def findDistance(self,p1,p2,img=None):
        x1,y1 = p1
        x2,y2 = p2
        cx,cy = (x1+x2)//2 ,(y1+y2)//2
        length = math.hypot(x2-x1,y2-y1)
        info = (x1,y1,x2,y2,cx,cy)
        if img is not None:
            cv2.circle(img , (x1,y1),2,(255,0,255),cv2.FILLED)
            cv2.circle(img, (x2, y2), 2, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.circle(img, (cx, cy), 2, (255, 0, 255), cv2.FILLED)
            return length , info ,img 
        else:
            return length , info 

def main():

    #实例化
    facedetector = FaceMeshDetector()

    cap = cv2.VideoCapture(0)

    while True:
        success,img = cap.read()
        if success:
            faces , img = facedetector.findFaceMesh(img)

            if faces:
                for face in faces:
                    print(len(face)) #468个点
                    leftEyeUpPoint = face[159]
                    leftEyeDownPoint = face[23]
                    leftEyeDistance ,info ,img = facedetector.findDistance(leftEyeUpPoint,leftEyeDownPoint,img)
                    print(leftEyeDistance)

            cv2.imshow('img',img)
            key = cv2.waitKey(1)
            if key==ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
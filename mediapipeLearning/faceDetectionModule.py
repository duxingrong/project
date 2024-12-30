import cv2 
import mediapipe as mp 

class FaceDetector():
    """
    检测人脸，显示画面和得到数据
    """
    def __init__(self,minDetectionCon =0.5,modelSelection=0):
        """
        :param minDetectionCon: 人脸检测成功的最小置信度值（范围：[0.0, 1.0])

        :param modelSelection: 0 或 1。选择 0 时，使用短距离模型，最佳效果适用于距离相机 2 米内的人脸；
        选择 1 时，使用全距离模型，最佳效果适用于距离相机 5 米内的人脸
        """
        self.minDetectionCon = minDetectionCon
        self.modelSelection = modelSelection
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(
            min_detection_confidence = self.minDetectionCon,
            model_selection = self.modelSelection
        )
    
    # 检测人脸
    def findFaces(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = [] #存放的所有人脸的字典
        h,w,c = img.shape
        if self.results.detections:
            for id , face in enumerate(self.results.detections):
                #如果检测的概率大于设定的值:
                if face.score[0]>self.minDetectionCon:
                    bboxC = face.location_data.relative_bounding_box
                    bbox = int(bboxC.xmin*w),int(bboxC.ymin*h), int(bboxC.width*w),int(bboxC.height*h)
                    cx,cy = bbox[0] + bbox[2]//2, bbox[1]+bbox[3]//2
                    bboxinfo ={'id':id,'bbox':bbox,'score':face.score,'center':(cx,cy)}
                    bboxs.append(bboxinfo)
                
                    #框起来并且加上置信率
                    if draw: 
                        self.cornerRect(img,bbox)
                        self.putTextRect(img,f'{int(face.score[0]*100)}%',(bbox[0],bbox[1]-10))


        return bboxs,img
                



    def cornerRect(self,img , bbox , l=30 ,t=5,rt=1,colorR=(255,0,255),colorC=(0,255,0)):
        """
        :param img: Image to draw on.
        :param bbox: Bounding box [x, y, w, h]
        :param l: length of the corner line
        :param t: thickness of the corner line
        :param rt: thickness of the rectangle
        :param colorR: Color of the Rectangle
        :param colorC: Color of the Corners
        :return:
        """
        x,y,w,h = bbox
        x1,y1 = x+w , y+h
        if rt!=0:
            cv2.rectangle(img , (x,y),(x1,y1),colorR,rt)
        
            # 开始加粗四个角
            ## Top Left x,y 
            cv2.line(img ,(x,y),(x+l,y),colorC,t)
            cv2.line(img ,(x,y),(x,y+l),colorC,t)
            ## Top Right x1,y 
            cv2.line(img ,(x1,y),(x1-l,y),colorC,t)
            cv2.line(img ,(x1,y),(x1,y+l),colorC,t) 
            ## Bottom Left x,y1
            cv2.line(img ,(x,y1),(x+l,y1),colorC,t)
            cv2.line(img ,(x,y1),(x,y1-l),colorC,t)
            ## Bottom Right x1,y1
            cv2.line(img ,(x1,y1),(x1-l,y1),colorC,t)
            cv2.line(img ,(x1,y1),(x1,y1-l),colorC,t)     
        return img                

    def putTextRect(self,img, text, pos, scale=2, thickness=2, colorT=(255, 255, 255),
                colorR=(255, 0, 255), font=cv2.FONT_HERSHEY_PLAIN,
                offset=5, border=None, colorB=(0, 255, 0)):
        """
        img: 输入图像，是需要在其上绘制文本和矩形的图像。
        text: 要绘制的文本字符串。
        pos: 文本的起始位置 (x1, y1)，即矩形左上角的坐标。
        scale: 文本的缩放比例，用于控制字体的大小。默认为 3。
        thickness: 文本的粗细。默认为 3。
        colorT: 文本的颜色，默认为白色 (255, 255, 255)。
        colorR: 矩形背景的颜色，默认为紫色 (255, 0, 255)。
        font: 字体类型，使用 OpenCV 定义的字体。默认为 cv2.FONT_HERSHEY_PLAIN。
        offset: 文本四周的边距，即矩形和文本之间的空间。默认为 10。
        border: 矩形的边框宽度。如果设置为 None,则没有边框。若设置为非 None 的值，则矩形有边框，宽度为该值。
        colorB: 矩形边框的颜色，默认为绿色 (0, 255, 0)。
        """
        ox,oy = pos 
        (w,h),_ = cv2.getTextSize(text,font,scale,thickness)

        # 根据字体大小加上间隔得到矩形的大小
        x1,y1,x2,y2 = ox-offset,oy+offset,ox+w+offset,oy-h-offset
        cv2.rectangle(img,(x1,y1),(x2,y2),colorR,cv2.FILLED)
        #是否选择加粗边框
        if border is not None :
            cv2.rectangle(img, (x1, y1), (x2, y2), colorB, border)

        cv2.putText(img,text,(ox,oy),font,scale,colorT,thickness)

        return img , [x1,y1,x2,y2]
    

def main():
    #实例化
    facedetector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

    cap = cv2.VideoCapture(0)

    while True:
        success,img = cap.read()
        if success:
            # 得到数据和图像
            bboxs,img = facedetector.findFaces(img)

            if bboxs:
                for bbox in bboxs:
                    ## 获取数据
                    id = bbox['id']
                    x,y,w,h = bbox['bbox']
                    score = int(bbox['score'][0]*100)
                    center = bbox['center']
                    print(score)
        
            cv2.imshow('Image',img)
            key = cv2.waitKey(1)
            if key== ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()
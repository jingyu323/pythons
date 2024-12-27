import numpy as np
import cv2
if __name__ == '__main__':
    # 给视频路径，打开视频
    cap = cv2.VideoCapture(0) # 打开本机的摄像头
    face_detector = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
    head = cv2.imread('./head.jpg')
    while True:
        flag,frame = cap.read() # flag是否读取了图片
        if not flag:
            break
        gray = cv2.cvtColor(frame, code=cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=10)
        for x,y,w,h in faces:
            # cv2.rectangle(frame,
            #               pt1 = (x,y),
            #               pt2=(x+w,y+h),
            #               color=[0,0,255],thickness=2)
            head2 = cv2.resize(head, dsize=(w, h))
            # TOTO 将小狗轮廓画上去，去掉白色的外边界。
            frame[y:y+h,x:x+w] = head2
        cv2.imshow('face',frame)
        key = cv2.waitKey(1000//24) # 注意是整除//，时间是毫秒，最小1毫秒，必须是整数
        if key == ord('q'): # 键盘输入q退出
            break
    cv2.destroyAllWindows()
    cap.release()
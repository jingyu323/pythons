import cv2
v = cv2.VideoCapture(0)
h_ = v.get(propId=cv2.CAP_PROP_FRAME_HEIGHT)
w_ = v.get(propId=cv2.CAP_PROP_FRAME_WIDTH)
print(h_,w_)
face_detector = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
head = cv2.imread('./head.jpg')
while True: # 死循环
    flag,frame = v.read()
    # 摄像头出问题，没有图片了，返回False
    if flag == False:
        break # 退出
    frame = cv2.resize(frame,dsize=(int(w_//2),int(h_//2)))
    gray = cv2.cvtColor(frame, code=cv2.COLOR_BGR2GRAY)
    # 播放慢了，检测人脸耗时操作，扫描整张图片，图片大，耗时长
    faces = face_detector.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=3)

    for x,y,w,h in faces:
        head2 = cv2.resize(head,dsize=(w,h))
        for i in range(h): # 高度，纵坐标
            for j in range(w): # w宽度，横坐标
                if (head2[i,j] >200).all():
                    pass
                else:
                    frame[i+y,j + x] = head2[i,j] # 第一维高度，第二维宽度
    cv2.imshow('frame',frame)
    key = cv2.waitKey(10) # 等待键盘输入的Key 键盘
    if key == ord('q'):
        break
cv2.destroyAllWindows()
v.release() # 释放，视频流
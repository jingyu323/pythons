import cv2

import cv2 as cv

num = 1
cap = cv.VideoCapture(0)

while (cap.isOpened()):#检测摄像头是否开启
    ret,frame = cap.read()#读取每一针的数据
#设置显示大小
    farm=cv.resize(frame,dsize=(1080,1080))
#显示图像
    cv.imshow("2",farm)
    cv.waitKey(1) & 0xFF#键盘检测
    #按键判断
    if cv.waitKey(1) & 0xFF == ord("s"):#键盘判断
#对图片进行保存
        cv.imwrite(r"D:\xuexi\python\pythonProject2\人脸识别"+str(num)+".jpg",farm)
        print("图片保存成功")
        num += 1
#设置按键退出程序
    elif cv.waitKey(1) & 0xFF == ord(" "):
        break


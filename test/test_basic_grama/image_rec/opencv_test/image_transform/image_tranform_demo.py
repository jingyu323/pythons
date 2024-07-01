import cv2
import numpy as np


def mouse_event(event, x, y, flags, param ):
    img=param[0]
    img_hsv=param[1]
    if event == cv2.EVENT_LBUTTONDBLCLK:  # 双击左键显示图像的坐标和对应的rgb值
        print('img pixel value at(', x, ',', y, '):', img[y, x])  # 坐标(x,y)对应的像素值应该是img[y,x]
        text = '(' + str(x) + ',' + str(y) + ')' + str(img[y, x])
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # 绘制文字

    if event == cv2.EVENT_RBUTTONDBLCLK:  # 双击右键显示图像的坐标和对应的hsv值
        print('img_hsv pixel value at(', x, ',', y, '):', img_hsv[y, x])  # 坐标(x,y)对应的像素值应该是img_hsv[y,x]
        text = '(' + str(x) + ',' + str(y) + ')' + str(img_hsv[y, x])
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # 绘制文字

def  house_img_check():
     img = cv2.imread('../image/house.png')
     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  #转换到hsv空间

     cv2.namedWindow('image', cv2.WINDOW_NORMAL)  # 定义窗口
     # cv2.resizeWindow('image',(800,800))

     cv2.setMouseCallback('image', mouse_event,[img,img_hsv])  # 鼠标回调

     while True:
         cv2.imshow('image', img)
         if cv2.waitKey(1) == ord('q'):
             break
     cv2.destroyAllWindows()


def book_trans_form():
    img_book = cv2.imread('../image/bookf.png')
    h, w, c = img_book.shape

    src = np.float32([[308, 207], [62, 424], [304, 617], [522, 319]])  # 用微信截图来确定roi区域的四个角的坐标点位置(或者用鼠标事件输出四角坐标)
    # dst = np.float32([[300, 200], [300, 500], [510, 500], [510, 200]])
    dst = np.float32([[0,0],[0,300],[210,300],[210,0]])  #只保留roi区域，warpPerspective函数中尺寸设置为roi的宽高

    M = cv2.getPerspectiveTransform(src, dst)  # 透视变换矩阵
    print(M)

    # new_book = cv2.warpPerspective(img_book, M, (w, h))  # 透视变换  二维平面获得接近三维物体的视觉效果的算法
    new_book = cv2.warpPerspective(img_book,M,(210,300)) #透视变换  变换后的roi区域的宽高，只保留roi区域

    cv2.imshow('img_book', img_book)
    cv2.imshow('new_book', new_book)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def change_ad_img():
    img2 = cv2.imread('../image/house.png')
    img1 = cv2.imread('../image/guagnb.jpg')
    h1, w1, c1 = img1.shape

    src = np.float32([[0, 0], [0, h1 - 1], [w1 - 1, h1 - 1], [w1 - 1, 0]])  # 需要嵌入的原图片的四角坐标（img1） 逆时针坐标
    dst = np.float32([[438, 53], [436, 397], [817, 528], [808, 378]])  # 待嵌入图片区域的位置坐标（img2）
    M = cv2.getPerspectiveTransform(src, dst)  # 透视变换矩阵

    h2, w2, c2 = img2.shape
    new_img1 = cv2.warpPerspective(img1, M, (w2, h2))  # 透视变换  二维平面获得接近三维物体的视觉效果的算法
    dst = dst.astype(int)  # 多边形的坐标需要整型

    cv2.fillConvexPoly(img2, dst, [0, 0, 0])  # 用多边形填充的办法把嵌入区域的像素全部变成0
    new_img = cv2.add(img2, new_img1)  # 把变换后的图片插入到广告牌的位置
    cv2.imshow('img1', img1)
    cv2.imshow('new_img1', new_img1)
    cv2.imshow('new_img', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
     # house_img_check()
     # book_trans_form()
     change_ad_img()


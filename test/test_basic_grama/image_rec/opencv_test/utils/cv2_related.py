import cv2
import imutils
import numpy as np


def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# 截取图像
def img_extract(img):

    result = img[0:200, 0:200]
    cv_show('result', result)
    print('result', img.shape)
    print('像素点个数',  img.size)
    print('数据类型',  img.dtype)

    # 保存
    # cv2.imwrite('xxx.jpg', img)

def read_video(video_path):
    vc = cv2.VideoCapture(video_path)
    if vc.isOpened():
        open, frame = vc.read()
    else:
        open = False
    # 循环播放每一帧图像
    while open:
        ret, frame = vc.read()
        if frame is None:
            break
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('result', gray)
            if cv2.waitKey(10) & 0xFF == 27:  # 27为键盘上的退出键
                break
    vc.release()
    cv2.destroyAllWIndows()


def border_fill(img):
    top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)

    # 复制法，复制最边缘像素
    replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
    cv_show('replicate', replicate)
    # 反射法，对感兴趣的像素在两边复制，如：fedcba|abcdefgh|hgfedcb
    reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT)
    cv_show('reflect', reflect)
    # 反射法，以最边缘像素为轴对称，如：gfedcb|abcdefgh|gfedcba
    reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,
                                    borderType=cv2.BORDER_REFLECT_101)
    cv_show('reflect101', reflect101)

    # 外包法，如：cdefgh|abcdefgh|abcdefg
    wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_WRAP)
    cv_show('wrap', wrap)
    # 常量法，用常数填充
    constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_CONSTANT,
                                  value=0)

    cv_show('constant', constant)


def  img_yuzhi(img):
    img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 超过阈值部分取maxval(最大值)，否则取0
    ret,thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    cv_show('thresh1', thresh1)
    #   大于阈值的使用0表示，小于阈值的使用最大值表示
    ret,thresh2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)  # INV为inverse

    cv_show('thresh2', thresh2)
    # 大于阈值部分为阈值，其他不变
    ret,thresh3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
    cv_show('thresh3', thresh3)
    # 大于阈值的不变，其他为0
    ret,thresh4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
    cv_show('thresh4', thresh4)
    # 上面的反转
    ret,thresh5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)
    # 127为阈值，255为最大值
    cv_show('thresh5', thresh5)

def img_quzao(img):
    # 均值滤波，简单的平均卷积操作
    blur = cv2.blur(img, (3, 3))  # 用3*3的单位矩阵卷积(一般都是奇数矩阵)
    # 方框滤波,类似均值滤波，可以选择归一化
    box = cv2.boxFilter(img, -1, (3, 3), normalize=True)  # normalize=False容易越界
    # 高斯滤波，高斯模糊的卷积核数值满足高斯分布，相当于更重视
    aussian = cv2.GaussianBlur(img, 1)
    # 中值滤波,用中间值代替
    median = cv2.medianBlur(img, 5)

    # 显示所有
    res = np.hstack((blur, aussian, median))  # 水平显示三张图像
    res = np.vstack((blur, aussian, median))  # 垂直显示三张图像

def clos_open(img):
    # 开运算：先腐蚀后膨胀
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # 闭运算：先膨胀后腐蚀
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # 礼帽=原始输入-开运算结果
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

    # 黑帽=闭运算-原始输入
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    # 梯度=膨胀-腐蚀
    gradient = cv2.morphologyEx(img,cv2.MORPH_GARDIENT.kernel)
    # dst = cv2.Sobel(src, ddepth, dx, dy, ksize)

    # ddepth:图像深度
    # dx和dy分别表示水平和竖直方向
    # ksize是Sobel算子大小

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

    v = cv2.Canny(img, 50, 100)

    img = cv2.imread('car.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv_show(thresh, 'thresh')
    # 轮廓特征
    contours=[0]
    cnt = contours[0]
    epsilon = 0.1*cv2.arcLength(cnt,True)
    approx = cv2.apporxPolyDP(cnt,epsilon,True)
    # 边界矩形
    cnt = contours[3]
    x, y, w, h = cv2.boundingRect(cnt)
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 外接圆
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)


def check_rec(img):
    bye_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    ret, bye_binary = cv2.threshold(bye_gray, 100, 255, cv2.THRESH_BINARY)  # 二值化，高于100的为255，低于100的为0
    # 只能用二值化
    cnts = cv2.findContours(bye_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            print(screenCnt)
            image_contours = cv2.drawContours(img, [screenCnt], contourIdx=-1, color=(0, 255, 0), thickness=3)  # 绘制轮廓
            cv_show('image_contours', image_contours)







if __name__ == '__main__':
    img = cv2.imread("../image/lunkuo.png")
    # img_extract(img)

    check_rec(img)

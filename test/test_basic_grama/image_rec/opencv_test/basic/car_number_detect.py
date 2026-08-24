import cv2
import easyocr
import numpy as np
from pytesseract import pytesseract


def car_number_detect(img):
    car = cv2.imread(img)
    cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)
    car_gray = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)
    car_detect = cv2.CascadeClassifier("../xml/haarcascade_car_plate.xml")
    plates = car_detect.detectMultiScale(car_gray)

    for (x, y, w, h) in plates:
        cv2.rectangle(car, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('car', car)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def lpr(filename): # Licence Plate recognize 车牌识别
    # 1、图片加载和预处理
    img = cv2.imread(filename)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # 灰度化处理
    GaussianBlur_img = cv2.GaussianBlur(gray_img, (3, 3), sigmaX=0) # 高斯平滑，高斯模糊
    canny = cv2.Canny(GaussianBlur_img,150,255) # 轮廓检测
    ret, binary_img = cv2.threshold(canny, 127, 255, cv2.THRESH_OTSU) # 二值化操作

    # 2、形态学运算
    kernel = np.ones((5, 5), np.uint8)
    # 先闭运算将车牌数字部分连接，再开运算较小的部分去掉
    close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    open_img = cv2.morphologyEx(close_img, cv2.MORPH_OPEN, kernel)
    # 由于部分图像得到的轮廓边缘不整齐，因此再进行一次膨胀操作
    dilation_img = cv2.dilate(open_img, np.ones(shape = [5,5],dtype=np.uint8), iterations=3)

    # 3、获取轮廓
    contours, hierarchy = cv2.findContours(dilation_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = [] # 获取轮廓的矩形
    for c in contours:
        x = []
        y = []
        for point in c:
            y.append(point[0][0])
            x.append(point[0][1])
        r = [min(y), min(x), max(y), max(x)]
        rectangles.append(r)

    # 4、根据HSV颜色空间查找汽车上车牌位置
    dist_r = []
    max_mean = 0
    for r in rectangles:
        block = img[r[1]:r[3], r[0]:r[2]] # block块
        hsv = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)
        low = np.array([100, 43, 46])
        up = np.array([124, 255, 255])
        result = cv2.inRange(hsv, low, up)
        # 用计算均值的方式找蓝色最多的区块
        mean = np.mean(result)
        if mean > max_mean:
            max_mean = mean
            dist_r = r
    # 画出识别结果，由于之前多做了一次膨胀操作，导致矩形框稍大了一些，因此这里对于框架+3-3可以使框架更贴合车牌
    cv2.rectangle(img, (dist_r[0]+3, dist_r[1]), (dist_r[2]-3, dist_r[3]), (0, 255, 0), 2)
    cv2.imshow("lpr", img)
    cv2.waitKey(0)
# 主程序


#  识别精度有问题 还需要继续解决
def car_number_detect_2(img):
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(img.shape)
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 转换到hsv空间

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))  # 形态学结构元

    LowerBlue = np.array([100, 90, 90])  # 检测hsv的上下限（蓝色车牌）
    UpperBlue = np.array([140, 220, 90])

    # inRange 函数将颜色的值设置为 1，如果颜色存在于给定的颜色范围内，则设置为白色，如果颜色不存在于指定的颜色范围内，则设置为 0
    mask = cv2.inRange(HSV, LowerBlue, UpperBlue)  # 车牌mask

    dilate = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)  # 形态学膨胀和开操作把提取的蓝色点连接起来
    morph = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel, iterations=5)


    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 找车牌的轮廓，只找外轮廓就行

    print(len(contours))
    img_copy = img.copy()
    cv2.drawContours(img_copy, contours, -1, [0, 0, 255], 2)  # 把轮廓画出来

    roi_img= gray.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        print( x, y, w, h)
        if w > 100 :
            rect = cv2.boundingRect(contour)  # 用矩形把轮廓框出来（轮廓外接矩形）
            (x, y, w, h) = rect
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow('car', img)

            roi_img = gray[y:y + h, x:x + w]  # 提取车牌区域进行ocr识别

    _,roi_thresh = cv2.threshold(roi_img,10,255,cv2.THRESH_BINARY )
    cv2.imshow('roi', roi_img)
    open_img = cv2.morphologyEx(roi_thresh,cv2.MORPH_OPEN,kernel,iterations=2)  #适当的形态学操作提高识别率
    cv2.imshow('open_img',open_img)


    print(pytesseract.image_to_string(roi_img,  lang='chi_sim+eng', config='--psm 6 --oem 3 '))  # ocr识别

    reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)

    result = reader.readtext(roi_img)
    print(result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()






if __name__ == '__main__':
    image = "../../chepai.png"
    # car_number_detect(image);

    car_number_detect_2(image)
import cv2  # 答题卡识别
import numpy as np


def imgBrightness(img1, c, b):
    rows, cols = img1.shape
    blank = np.zeros([rows, cols], img1.dtype)
    rst = cv2.addWeighted(img1, c, blank, 1 - c, b)
    return rst

def red_img():
    img = cv2.imread('images/5.png')
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯滤波
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # 增强亮度
    blurred = imgBrightness(blurred, 1.5, 3)

    # 自适应二值化
    blurred = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 2)

    '''
    adaptiveThreshold函数：第一个参数src指原图像，原图像应该是灰度图。
        第二个参数x指当像素值高于（有时是小于）阈值时应该被赋予的新的像素值
        第三个参数adaptive_method 指： CV_ADAPTIVE_THRESH_MEAN_C 或 CV_ADAPTIVE_THRESH_GAUSSIAN_C
        第四个参数threshold_type  指取阈值类型：必须是下者之一  
                            • CV_THRESH_BINARY,
                            • CV_THRESH_BINARY_INV
        第五个参数 block_size 指用来计算阈值的象素邻域大小: 3, 5, 7, ...
        第六个参数param1    指与方法有关的参数。对方法CV_ADAPTIVE_THRESH_MEAN_C 和 CV_ADAPTIVE_THRESH_GAUSSIAN_C， 它是一个从均值或加权均值提取的常数, 尽管它可以是负数。
    '''
    blurred = cv2.copyMakeBorder(blurred, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))

## 图像轮廓
import cv2
import numpy as np
import matplotlib.pyplot as plt


def count_find():
    cat = cv2.imread('../image/lunkuo.png')

    gray = cv2.cvtColor(cat, cv2.COLOR_BGR2GRAY)

    #  返回两个值 一个是阈值 一个是 二值化之后的值

    thresh, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    #  层级  hier
    cou, hier = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(cou)

    copy = cat.copy()
    # -1   是所有  0 ， 1 是从外网内画轮廓
    cv2.drawContours(copy, cou, 1, (0, 0, 255), 2)

    area = cv2.contourArea(cou[1])

    print(area)

    perimeter = cv2.arcLength(cou[1], closed=True)

    print(perimeter)

    cv2.imshow("cat", cat)
    cv2.imshow("copy", copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def tu_bao():
    cat = cv2.imread('../image/hand.png')

    gray = cv2.cvtColor(cat, cv2.COLOR_BGR2GRAY)

    #  返回两个值 一个是阈值 一个是 二值化之后的值

    thresh, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    #  层级  hier
    cou, hier = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # -1   是所有  0 ， 1 是从外网内画轮廓
    cv2.drawContours(cat, cou, 0, (0, 0, 255), 2)

    # 多边形逼近
    approx = cv2.approxPolyDP(cou[0], 20, True)
    # -1   是所有  0 ， 1 是从外网内画轮廓
    cv2.drawContours(cat, [approx], 0, (0, 255, 0), 2)
    # 计算凸包
    hull = cv2.convexHull(cou[0])
    cv2.drawContours(cat, [hull], 0, (255, 0, 0), 2)
    cv2.imshow("cat", cat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 最大外接矩形
def max_rec():
    cat = cv2.imread('../image/hello.png')

    gray = cv2.cvtColor(cat, cv2.COLOR_BGR2GRAY)

    #  返回两个值 一个是阈值 一个是 二值化之后的值

    thresh, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    #  层级  hier
    cou, hier = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # -1   是所有  0 ， 1 是从外网内画轮廓
    # cv2.drawContours(cat,cou,1, (0,0,255),2)
    rec = cv2.minAreaRect(cou[1])

    ## 计算旋转坐标
    box = cv2.boxPoints(rec)
    box = np.round(box).astype('int64')
    # 最小外接矩形
    cv2.drawContours(cat, [box], 0, (255, 0, 0), 2)
    # 最大外接矩形
    x, y, w, h = cv2.boundingRect(cou[1])
    cv2.rectangle(cat, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("cat", cat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 金字塔
def pyramid():
    cat = cv2.imread('../image/paper.png')
    dst = cv2.pyrDown(cat)
    cv2.imshow("dst", dst)
    cv2.imshow("cat", cat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pyramid_up():
    cat = cv2.imread('../image/paper.png')
    dst = cv2.pyrUp(cat)
    cv2.imshow("dst", dst)
    cv2.imshow("cat", cat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#  拉普拉斯金字塔用于图片压缩


#  直方图
def hits():
    cat = cv2.imread('../image/paper.png')

    gray = cv2.cvtColor(cat, cv2.COLOR_BGR2GRAY)

    print(gray)
    plt.hist(gray.ravel(), bins=256, range=[0, 255])
    plt.show()
    # hits = cv2.calcHist([cat],[0],None,[256],[0,255])

    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


#  直方图
def calsc_ts():
    cat = cv2.imread('../image/paper.png')

    hitb = cv2.calcHist([cat], [0], None, [256], [0, 255])
    hitg = cv2.calcHist([cat], [1], None, [256], [0, 255])
    hitr = cv2.calcHist([cat], [2], None, [256], [0, 255])

    plt.plot(hitb, color='b', label='blue')
    plt.plot(hitg, color='g', label='green')
    plt.plot(hitr, color='r', label='red')
    plt.show()


#  直方图
def yanmo():
    cat = cv2.imread('../image/paper.png')
    gray = cv2.cvtColor(cat, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, np.uint8)

    #  设置抠图区域
    mask[200:400, 200:400] = 255

    hist_mask = cv2.calcHist([cat], [0], mask, [256], [0, 255])
    hist_gray = cv2.calcHist([cat], [0], None, [256], [0, 255])
    plt.plot(hist_mask, color='b', label='hist_mask')
    plt.plot(hist_gray, color='g', label='hist_gray')
    # plt.show()
    cv2.imshow("mask", mask)
    cv2.imshow("gray", gray)
    cv2.imshow("mask_gray", cv2.bitwise_and(gray, gray, mask=mask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 一个函数来计算两张图片之间的均方误差
def mse(img1, img2):
    h, w = img1.shape
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff ** 2)
    mse = err / (float(h * w))
    return mse


#  用方差来比较图片是否一致
def comapare_image():
    # 加载输入图片
    img1 = cv2.imread('panda.jpg')
    img2 = cv2.imread('panda1.jpg')
    img3 = cv2.imread('panda1.jpg')

    # 将图片转换为灰度图
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 计算三幅图片之间的MSE误差
    error1, diff1 = mse(img1, img2)
    error2, diff2 = mse(img2, img3)
    error3, diff3 = mse(img1, img3)

    # 输出MSE误差值
    print("Image matching Error between the two images:", error1, error2, error3)

    # 展示三幅图片间的区别
    cv2.imshow("difference between image 1 and 2", diff1)
    cv2.imshow("difference between image 2 and 3", diff2)
    cv2.imshow("difference between image 1 and 3", diff3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def zhifang():
    # 读取输入图像
    img = cv2.imread('../image/')

    # 将图像分割成各自的蓝色、绿色和红色通道
    blue, green, red = cv2.split(img)

    # 绿色和蓝色通道的2D颜色直方图
    plt.subplot(131)
    hist1 = cv2.calcHist([green, blue], [0, 1], None, [32, 32], [0, 256, 0, 256])
    p = plt.imshow(hist1, interpolation="nearest")
    plt.title("G和B的2D直方图", fontsize=8)
    plt.colorbar(p)

    # 红色和绿色通道的2D颜色直方图
    plt.subplot(132)
    hist2 = cv2.calcHist([red, green], [0, 1], None, [32, 32], [0, 256, 0, 256])
    p = plt.imshow(hist2, interpolation="nearest")
    plt.title("R和G的2D直方图", fontsize=8)
    plt.colorbar(p)

    # 蓝色和红色通道的2D颜色直方图
    plt.subplot(133)
    hist3 = cv2.calcHist([blue, red], [0, 1], None, [32, 32], [0, 256, 0, 256])
    p = plt.imshow(hist3, interpolation="nearest")
    plt.title("B和R的2D直方图", fontsize=8)
    plt.colorbar(p)
    plt.show()


# 模板匹配方法 识别图像
def template_match():
    img_rgb = cv2.imread('../image/jin_zhen.png')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('../image/jinc.png', 0)
    h, w = template.shape[:2]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    # 取匹配程度大于%80的坐标
    loc = np.where(res >= threshold)
    print(loc)
    # np.where返回的坐标值(x,y)是(h,w)，注意h,w的顺序
    for pt in zip(*loc[::-1]):
        print(pt)
        bottom_right = (pt[0] + w, pt[1] + h)
        cv2.rectangle(img_rgb, pt, bottom_right, (0, 0, 255), 2)

    cv2.imshow('img_rgb', img_rgb)
    cv2.waitKey(0)


# 均衡器
def junheng():
    img = cv2.imread('../image/nvpai.jpeg', 0)
    # 均衡化
    equ = cv2.equalizeHist(img)
    plt.hist(equ.ravel(), bins=256, range=[0, 255])
    # 自适应均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    res_clahe = clahe.apply(img)
    res = np.hstack((img, res_clahe, equ))
    cv2.imshow('img', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def bonder_test():
    bye1 = cv2.imread('../image/bye.png')
    bye = cv2.resize(bye1, (400, 500))

    bye_gray = cv2.cvtColor(bye, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    ret, bye_binary = cv2.threshold(bye_gray, 100, 255, cv2.THRESH_BINARY)  # 二值化，高于100的为255，低于100的为0
    contours, hierarchy = cv2.findContours(bye_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 获取轮廓，使用RETR_EXTERNAL方法只获取外轮廓
    epsilon = 0.001 * cv2.arcLength(contours[1], True)  # 设置近似精度，此处选取第一条轮廓，设为其周长的0.01倍  精度越小轮廓约精细 越准确
    approx = cv2.approxPolyDP(contours[1], epsilon, True)  # 对轮廓进行近似
    bye_new = bye.copy()  # 复制一份，不破坏原图

    image_contours = cv2.drawContours(bye_new, [approx], contourIdx=-1, color=(0, 255, 0), thickness=3)  # 绘制轮廓
    cv2.imshow('bye', bye)
    cv2.waitKey(100000)
    cv2.imshow('image_contours', image_contours)
    cv2.waitKey(100000)


if __name__ == '__main__':
    bonder_test()

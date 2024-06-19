import cv2  # 答题卡识别
import numpy as np

from opencv_test.utils.cv2_related import cv_show


def imgBrightness(img1, c, b):
    rows, cols = img1.shape
    blank = np.zeros([rows, cols], img1.dtype)
    rst = cv2.addWeighted(img1, c, blank, 1 - c, b)
    return rst


def order_points(pts):
    # 初始化坐标点
    rect = np.zeros((4, 2), dtype="float32")

    # 获取左上角和右下角坐标点
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 分别计算左上角和右下角的离散差值
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    # 获取坐标点，并将它们分离开来
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算新图片的宽度值，选取水平差值的最大值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # 计算新图片的高度值，选取垂直差值的最大值
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 构建新图片的4个坐标点
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 获取仿射变换矩阵并应用它
    M = cv2.getPerspectiveTransform(rect, dst)
    # 进行仿射变换
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # 返回变换后的结果
    return warped
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

    edged = cv2.Canny(blurred, 0, 255)
    cnts, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    docCnt = []
    count = 0
    # 确保至少有一个轮廓被找到
    if len(cnts) > 0:
        # 将轮廓按照大小排序
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # 对排序后的轮廓进行循环处理
    for c in cnts:
        # 获取近似的轮廓
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # 如果近似轮廓有四个顶点，那么就认为找到了答题卡
        if len(approx) == 4:
            docCnt.append(approx)
            count += 1
            if count == 3:
                break
    #四点变换，划出选择题区域
    paper = four_point_transform(img,np.array(docCnt[0]).reshape(4,2))
    warped = four_point_transform(gray,np.array(docCnt[0]).reshape(4,2))
    #四点变换，划出准考证区域
    ID_Area = four_point_transform(img,np.array(docCnt[1]).reshape(4,2))
    ID_Area_warped = four_point_transform(gray,np.array(docCnt[1]).reshape(4,2))
    #四点变换，划出科目区域
    Subject_Area = four_point_transform(img,np.array(docCnt[2]).reshape(4,2))
    Subject_Area_warped = four_point_transform(gray,np.array(docCnt[2]).reshape(4,2))


def hellotransform():
    image = cv2.imread('../image/hello01.png')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, template_img = cv2.threshold(gray, 10, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    cnts,her=cv2.findContours(template_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    screenCnt=[]
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            print(screenCnt)
            image_contours = cv2.drawContours(image, [screenCnt], contourIdx=-1, color=(0, 255, 0), thickness=3)  # 绘制轮廓
            cv_show('image_contours', image_contours)

    print(type(screenCnt))
    print(screenCnt.reshape(4,2))

    # 获取原始的坐标点 ， 使用reshape 进行行列变换
    pts = np.array( screenCnt.reshape(4,2), dtype="float32")

    print(pts)

    # 对原始图片进行变换
    warped = four_point_transform(image, pts)

    # 结果显示
    cv2.imshow("Original", image)
    cv2.imshow("Warped", warped)
    cv2.waitKey(0)



if __name__ == '__main__':
    hellotransform()
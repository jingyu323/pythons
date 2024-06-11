# 霍夫变换是一种在图像中寻找直线、圆形以及其他简单形状的方法。霍夫变换采用类似于投票的方式来获取当前图像内的形状集合，该变换由
# Paul
# Hough（霍夫）于
# 1962
# 年首次提出。
# 最初的霍夫变换只能用于检测直线，经过发展后，霍夫变换不仅能够识别直线，还能识别其他简单的图形结构，常见的有圆、椭圆等。
# 霍夫变换应用场景
# 霍夫变换在计算机视觉和图像处理领域中具有广泛的应用，主要用于检测图像中的几何形状，特别是直线和圆。以下是一些霍夫变换的应用场景：
#
# 直线检测： 霍夫变换可用于检测图像中的直线，无论这些直线是水平、垂直还是倾斜的。应用场景包括道路标线检测、图像中的边缘检测、图像中的线条检测等。
#
# 圆检测： 圆霍夫变换可用于检测图像中的圆形，例如在图像中识别圆形物体、检测圆形物体的轮廓等。
#
# 椭圆检测： 扩展的霍夫变换可以用于检测图像中的椭圆，应用场景包括眼球检测、图像中的椭圆轮廓识别等。
#
# 直线和圆的参数估计： 霍夫变换可以用于估计图像中的直线和圆的参数，例如直线的斜率和截距，圆的半径和圆心位置。
#
# 图像分析与特征提取： 霍夫变换可以用于分析图像中的几何特征，例如检测道路交叉口、分析图像中的纹理等。
#
# 形状检测与识别： 霍夫变换可以用于识别特定形状的对象，例如在工业检测中检测零件的缺陷。
#
# 图像中的直线和轮廓重建： 通过霍夫变换，可以检测到图像中的直线和轮廓，然后利用这些信息进行图像重建。
#
# 图像配准与拼接： 霍夫变换可用于在不同视角下或不同图像之间检测共享几何结构，从而实现图像配准和拼接。
#
# 数字图像处理和计算机视觉教学： 霍夫变换是一个重要的教学工具，用于讲解图像处理中的几何形状检测原理。
#
import cv2
import numpy as np

# Hough直线变换，可以检测一张图像中的直线
def hough_test():
    img = cv2.imread('../image/qipan.png', cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    for i in range(len(lines)):
        # for rho, thetha in lines[10]:
        rho = lines[i][0][0]
        thetha = lines[i][0][1]
        a = np.cos(thetha)
        b = np.sin(thetha)
        x0 = a * rho
        y0 = b * rho
        line_length = 1000  # 线长
        x1 = int(x0 + line_length * (-b))
        y1 = int(y0 + line_length * (a))
        x2 = int(x0 - line_length * (-b))
        y2 = int(y0 - line_length * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # 因为gray和edges都是单通道的，为了可以和原图拼接合并，需要merge成3通道图像数据
    gray = cv2.merge((gray, gray, gray))
    edges = cv2.merge((edges, edges, edges))

    # 图像拼接
    res = np.hstack((gray, edges, img))

    cv2.imshow('res', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 渐进概率式霍夫变换
# cv2.HoughLinesP(image: Mat, rho, theta, threshold, lines=…, minLineLength=…, maxLineGap=…)

def houghp_test():
    img = cv2.imread('../image/qipan.png', cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50, 150, apertureSize=3)

    minLineLength = 100
    maxLineGap = 10
    # HoughLinesP(image: Mat, rho, theta, threshold, lines=..., minLineLength=..., maxLineGap=...)
    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 100, minLineLength, maxLineGap)

    print(lines.shape)
    print(lines[0])

    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    gray = cv2.merge((gray, gray, gray))
    canny = cv2.merge((canny, canny, canny))

    res = np.hstack((gray, canny, img))
    cv2.imshow('res', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':

    houghp_test()

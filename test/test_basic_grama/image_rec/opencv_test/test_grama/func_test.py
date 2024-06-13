# 多边形   cv2.polylines()用来画多边形。
import cv2
import numpy as np


def polylines_test():
    img = np.ones((512, 512, 3))  # 白色背景
    color = (0, 255, 0)  # 绿色
    # 五角星
    pts = np.array([[70, 190], [222, 190], [280, 61], [330, 190], [467, 190],
                    [358, 260], [392, 380], [280, 308], [138, 380], [195, 260]])
    print(pts)
    pts = pts.reshape((-1, 1, 2))  # reshape为10x1x2的numpy
    print(pts.shape)  # (10, 1, 2)
    print(pts)
    cv2.polylines(img, [pts], True, color, 5)
    # 方形
    pts = np.array([[10, 50], [100, 50], [100, 100], [10, 100]])
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, color, 3)
    cv2.fillPoly(img,  # 原图画板
                 [pts],  # 多边形的点
                 color=(0, 0, 255))
    cv2.imshow('juzicode.com', img)
    cv2.waitKey()


if __name__ == '__main__':

    polylines_test()


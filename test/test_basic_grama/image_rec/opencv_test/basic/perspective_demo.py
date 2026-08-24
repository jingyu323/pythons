from random import random

import cv2
import numpy as np
from matplotlib import pyplot as plt


def warpperspective_demo():
    h, w = 200, 300  # 图像大小
    origin_image = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([[50, 50], [100, 160], [250, 100]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    origin_image = cv2.fillConvexPoly(origin_image, pts, 255, 100)
    # 透视变换矩阵   M： 变换矩阵，即3x3的透视变换矩阵。
    M = np.array([[0.99, -0.11, 80], [-0.11, 0.99, 30], [0, 0, 1]])
    # opencv透视变换
    dst_opencv = cv2.warpPerspective(origin_image.copy(), M, (w, h))
    # 显示原始图像和经过透视变换后的图像
    cv2.imshow('original', origin_image)
    cv2.imshow('opencv', dst_opencv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def get_points_tran(points_list, M):
    '''透视变换坐标转换'''
    for i in points_list:
        i[0],i[1] = cvt_pos([i[0],i[1]],M)
        i[2],i[3] = cvt_pos([i[2],i[3]],M)
        i[4],i[5] = cvt_pos([i[4],i[5]],M)
        i[6],i[7] = cvt_pos([i[6],i[7]],M)
    return points_list


def cvt_pos(pos,cvt_mat_t):
    u = pos[0]
    v = pos[1]
    x = (cvt_mat_t[0][0]*u+cvt_mat_t[0][1]*v+cvt_mat_t[0][2])/(cvt_mat_t[2][0]*u+cvt_mat_t[2][1]*v+cvt_mat_t[2][2])
    y = (cvt_mat_t[1][0]*u+cvt_mat_t[1][1]*v+cvt_mat_t[1][2])/(cvt_mat_t[2][0]*u+cvt_mat_t[2][1]*v+cvt_mat_t[2][2])
    return (int(x),int(y))

def perspective_transformation(Image, total_points_list):
    h, w, ch = Image.shape  # 获取行数（高）和列数（宽）
    # 原图四个角坐标
    p1 = np1 = [0, 0]  # 左上
    p2 = np2 = [w - 1, 0]  # 右上
    p3 = np3 = [w - 1, h - 1]  # 右下
    p4 = np4 = [0, h - 1]  # 左下

    pts1 = np.float32([p1, p2, p3, p4])
    np_list = [np1, np2, np3, np4]

    for i in range(0, random.randint(1, 4)):
        np_list[i][0] = np_list[i][0] + random.randint(0, 50)
        np_list[i][1] = np_list[i][1] + random.randint(0, 50)

        # print('指定透视变换后四角坐标:%s' % np_list)
        nw = max(np1[0], np2[0], np3[0], np4[0])
        nh = max(np1[1], np2[1], np3[1], np4[1])
        pts2 = np.float32([np1, np2, np3, np4])
        # 获得透视变换矩阵
        M = cv2.getPerspectiveTransform(pts1, pts2)
        # print('M:%s' %M) # 3行3列
        # 应用
        dst = cv2.warpPerspective(Image, M, (nw, nh))
        total_points_list = get_points_tran(total_points_list, M)
        # print('变换后的四角坐标:%s' % get_points_tran([p1+p2+p3+p4], M))
        return dst, total_points_list


def  spectiveTransform_demo( ):
    image = cv2.imread('../myimages/5.jpg')
    # 图片原来的固有字段的坐标及值
    points_list = []
    with open('../myimages/gt_5.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            # 获取前8个元素，也就是四点坐标，后面的是文本内容，可以不用变
            data_line = line.split(',')
            # 将列表中坐标元素变成int
            data = [int(i) if data_line.index(i) <= 7 else i for i in data_line]
            points_list.append(data)
    print(points_list)
    # 调用透视变换函数
    new_image, points_list = perspective_transformation(image, points_list)
    # 将新的图片写在本地
    cv2.imwrite('new_image5.jpg', new_image)

    # 也可以在窗口查看效果
    plt.subplot(121), plt.imshow(image), plt.title('Input')
    plt.subplot(122), plt.imshow(new_image), plt.title('Output')
    plt.show()


if __name__ == '__main__':
    spectiveTransform_demo()
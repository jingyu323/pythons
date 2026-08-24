import cv2
import numpy as np

# https://download.csdn.net/blog/column/12561987/136090893

#  mask image 未找到 效果没是成功

#  智能修复老照片
# https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life
def  linea_insert_demo():


    # 读取原始图像
    image = cv2.imread('../image/r1.png')

    # 设置放大倍数
    scale_percent = 2  # 放大两倍

    # 计算放大后的图像尺寸
    width = int(image.shape[1] * scale_percent)
    height = int(image.shape[0] * scale_percent)
    dim = (width, height)

    # 使用最近邻插值方法进行放大
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_NEAREST)

    # 显示原始图像和放大后的图像
    cv2.imshow('Original Image', image)
    cv2.imshow('Resized Image', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def  r2_demo():
    # 读取原始图像
    image = cv2.imread('../image/r2.png')
    # 读取原始图像和掩码图像
    mask = cv2.imread('mask_image.jpg', 0)  # 灰度图像作为掩码，缺失区域为255，非缺失区域为0

    # 使用纹理合成修复图像
    inpaint_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)  # 第三个参数为修复半径，第四个参数为修复方法

    # 显示原始图像、掩码图像和修复后的图像
    cv2.imshow('Original Image', image)
    cv2.imshow('Mask Image', mask)
    cv2.imshow('Inpainted Image', inpaint_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def  r4_demo():
    # 读取原始图像
    image = cv2.imread('../image/r4.png')

    # mask = cv2.imread('mask_image.jpg', 0)  # 灰度图像作为掩码，缺失区域为255，非缺失区域为0
    mask = np.ones(image.shape, np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # 使用边缘保持的方法进行图像修复
    edges = cv2.Canny(image, 100, 200)  # 提取原始图像的边缘信息
    inpaint_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)  # 第三个参数为修复半径，第四个参数为修复方法
    inpaint_image = cv2.bitwise_and(inpaint_image, inpaint_image, mask=edges)  # 保持边缘的连续性和一致性

    # 显示原始图像、掩码图像和修复后的图像
    cv2.imshow('Original Image', image)
    cv2.imshow('Mask Image', mask)
    cv2.imshow('Inpainted Image', inpaint_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # linea_insert_demo()
    r4_demo()

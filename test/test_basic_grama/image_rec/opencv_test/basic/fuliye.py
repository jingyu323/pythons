"""
分析不同滤波的特性
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../image/moon.png', 0)
def demo1():

    # 转化为32位格式
    img_float32 = np.float32(img)
    # 将其从左上角移动到中心位置
    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    # cv2.magnitude(dft_shift[:,:,0],(dft_shift)[:,:,1]) 对两个通道进行变化，将其变回图像，20*是映射到0-255之间
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], (dft_shift)[:, :, 1]))
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('magnitude_spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()


def  low_demo():
    # 转化为32位格式
    img_float32 = np.float32(img)
    # 将其从左上角移动到中心位置
    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    # mask处理测定中心位置
    # 并且转化为int值
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    # 低通滤波
    # 只有中心位置是1，其余位置都是0
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
    # IDFT 变回原始图像
    # fshift=dft_shift*mask 将DFT源码与mask结合，是一的保留，不是一的过滤掉了
    # f_ishift = np.fft.ifftshift(fshift) shift的负变化，将坐标一直左上角
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    # 实部虚部转换进行处理
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_back, cmap='gray')
    plt.title('Result'), plt.xticks([]), plt.yticks([])
    # 展示结果
    plt.show()

def hight_demo():
    # 高通高通滤波器
    # 转化为32位格式
    img_float32 = np.float32(img)
    # 将其从左上角移动到中心位置
    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    # mask处理测定中心位置
    # 并且转化为int值
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    # 低通滤波
    # 只有中心位置是1，其余位置都是0
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
    # IDFT 变回原始图像
    # fshift=dft_shift*mask 将DFT源码与mask结合，是一的保留，不是一的过滤掉了
    # f_ishift = np.fft.ifftshift(fshift) shift的负变化，将坐标一直左上角
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    # 实部虚部转换进行处理
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_back, cmap='gray')
    plt.title('Result'), plt.xticks([]), plt.yticks([])
    # 展示结果
    plt.show()

if __name__ == '__main__':
     hight_demo()
import glob

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import gridspec


def demo():
    plt.figure(num=1, figsize=(4, 4))
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    plt.plot(x, y)
    plt.show()


def demo1():
    fig = plt.figure(num=1, figsize=(4, 4))
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    ax = fig.add_subplot()
    ax.plot(x, y)
    plt.show()


def demo2():
    fig = plt.figure(num=1, figsize=(1, 1))
    ax1 = fig.add_subplot(221)  # 劃分2行2列區域，取第1個
    ax1.plot([1, 2, 3, 4], [1, 2, 3, 4])
    ax2 = fig.add_subplot(223)  # 劃分2行3列區域，取第3個
    ax2.plot([1, 2, 3, 4], [2, 2, 3, 4])
    ax3 = fig.add_subplot(222)  # 劃分3行2列區域，取第5個
    ax3.plot([1, 2, 3, 4, 5], [1, 2, 2, 4, 6])
    ax4 = fig.add_subplot(224)  # 劃分2行2列區域，取第4個
    ax4.plot([1, 2, 3, 4], [1, 2, 3, 3])
    plt.show()


def demo3():
    fig = plt.figure(num=1, figsize=(9, 9))
    img_path = '../Keras/cat_dog/image/superman/*'
    name_list = glob.glob(img_path)
    print(name_list)

    for i in range(3):
        img = Image.open(name_list[i])
        aix = plt.subplot(221 + i)
        aix.imshow(img)
    plt.show()


# 多行多列
def demo4():
    fig = plt.figure(num=1, figsize=(9, 9))
    ax1 = fig.add_subplot(11, 9, 1)  # 劃分11行9列區域，取第1個
    ax1.plot([1, 2, 3, 4], [1, 2, 3, 4])
    ax2 = fig.add_subplot(2, 3, 3)  # 劃分2行3列區域，取第3個
    ax2.plot([1, 2, 3, 4], [2, 2, 3, 4])
    ax3 = fig.add_subplot(3, 2, 5)  # 劃分3行2列區域，取第5個
    ax3.plot([1, 2, 3, 4], [1, 2, 2, 4])
    ax4 = fig.add_subplot(2, 2, 4)  # 劃分2行2列區域，取第4個
    ax4.plot([1, 2, 3, 4], [1, 2, 3, 3])
    plt.show()


def demo5():
    fig = plt.figure(num=1, figsize=(6, 6))  # 创建画布
    gs = gridspec.GridSpec(3, 3)  # 设定3*3的网格，使用下标时从0开始
    ax1 = fig.add_subplot(gs[0, :])  # 第1行，所有列
    ax1.plot([1, 2, 3, 4], [1, 2, 3, 4])
    ax2 = fig.add_subplot(gs[1, :-1])  # 第2行，倒数第一列之前的所有列
    ax2.plot([1, 2, 3, 4], [1, 2, 3, 4])
    ax3 = fig.add_subplot(gs[1:, -1])  # 第2行之后的所有行，倒数第一列
    ax3.plot([1, 2, 3, 4], [1, 2, 3, 4])
    ax4 = fig.add_subplot(gs[2, 0])  # 第3行，第1列
    ax4.plot([1, 2, 3, 4], [1, 2, 3, 4])
    ax5 = fig.add_subplot(gs[2, 1])  # 第3行，第2列
    ax5.plot([1, 2, 3, 4], [1, 2, 3, 4])
    plt.show()


# 限制刻度范围，set_xlim()、set_ylim()
def demo6():
    fig = plt.figure(num=1, figsize=(6, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title('exp')  # 添加标题
    ax1.set_xlim(1, 7.1)  # x轴限制坐标从1到7.1
    ax1.set_ylim(-10, 10)  # y轴限制坐标从-10到10
    ax1.plot([1, 2, 3, 4], [1, 2, 3, 4])
    plt.show()


def demo7():
    fig = plt.figure(num=1, figsize=(9, 9))
    img_path = '../Keras/cat_dog/image/superman/*'
    name_list = glob.glob(img_path)
    print(name_list)

    for i in range(3):
        img = Image.open(name_list[i])
        # aix = plt.subplot(221 + i)
        plt.subplot(1, 3, 1 + i)
        plt.imshow(img)
    plt.show()


def demo8():
    fig, axs = plt.subplots(2, 2, figsize=(20, 18))

    # 在每个子图中绘制一些数据
    for i in range(2):
        for j in range(2):
            x = np.linspace(0, 10, 100)
            y = np.sin(x + i + j)
            axs[i, j].plot(x, y)
            axs[i, j].set_title(f'Subplot {i + 1},{j + 1} - how2matplotlib.com')

    plt.tight_layout()
    plt.show()


def demo9():
    # 创建一个3x2的子图布局，并设置图形大小
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    # 在每个子图中绘制一些数据
    for i in range(3):
        for j in range(2):
            x = np.linspace(0, 5, 50)
            y = np.exp(-x) * np.sin(2 * np.pi * x + i * j)
            axs[i, j].plot(x, y)
            axs[i, j].set_title(f'Subplot {i + 1},{j + 1} - how2matplotlib.com')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    demo9()

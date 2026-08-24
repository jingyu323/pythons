import cv2
import numpy as np
from PIL import Image
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

"""
class_mode: "categorical", "binary", "sparse"或None之一. 默认为"categorical. 该参数决定了返回的标签数组的形式,"categorical"会返回2D的one-hot编码标签,
"binary"返回1D的二值标签."sparse"返回1D的整数标签,如果为None则不返回任何标签, 生成器将仅仅生成batch数据, 
这种情况在使用model.predict_generator()和model.evaluate_generator()等函数时会用到.
"""


def demo():
    datagen = ImageDataGenerator()

    # 设置训练图像的路径
    train_dir = 'E:/data/kreas/Kaggle/cat-dog-small/train'

    # 设置验证图像的路径
    validation_dir = 'E:/data/kreas/Kaggle/cat-dog-small/test'
    # 使用flow_from_directory方法加载训练图像
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),  # 图像尺寸
        batch_size=32,  # 批量大小
        class_mode='categorical'  # 分类模式
    )

    # 使用flow_from_directory方法加载验证图像
    validation_generator = datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),  # 图像尺寸
        batch_size=32,  # 批量大小
        class_mode='categorical'  # 分类模式
    )

    # 获取训练图像的数量
    train_image_count = train_generator.samples

    print(train_image_count)


# 图片水平反转
def image_change():
    image_path = 'bare.png'
    img = Image.open(image_path)
    horizontal_flipped = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    horizontal_flipped.show()
    # cv2.imshow("Image", horizontal_flipped)
    # print("dddd")
    # cv2.waitKey(0)


def image_vertical_flip():
    image_path = 'bare.png'
    img = Image.open(image_path)

    # 垂直翻转
    vertical_flipped = img.transpose(method=Image.FLIP_TOP_BOTTOM)

    # 保存翻转后的图像
    vertical_flipped.save('vertical_flipped_image.png')

    # 显示翻转后的图像 (可选)
    vertical_flipped.show()


def demo_img():
    # 读取图像
    img1 = cv2.imread("bare.png")
    # 图像水平镜像
    h, w = img1.shape[:2]
    M = np.float32([[-1, 0, 500], [0, 1, 0]])
    dst1 = cv2.warpAffine(img1, M, (w, h))

    # 图像垂直镜像
    h, w = img1.shape[:2]
    M = np.float32([[1, 0, 0], [0, -1, 500]])
    dst2 = cv2.warpAffine(img1, M, (w, h))

    # 图像显示
    fig, axes = plt.subplots(1, 3, figsize=(80, 10), dpi=100)
    axes[0].imshow(img1[:, :, ::-1])
    axes[0].set_title("original")
    axes[1].imshow(dst1[:, :, ::-1])
    axes[1].set_title("after horizontal-mirror")
    axes[2].imshow(dst2[:, :, ::-1])
    axes[2].set_title("after vertical-mirror")
    plt.show()


def demo1_image():
    # 图像垂直镜像
    img1 = cv2.imread("bare.png")
    img2 = img1.copy()
    img3 = img1.copy()

    # 图像水平镜像
    RM1 = np.float32([[1, 0, 0], [0, -1, 500], [0, 0, 1]])  # 计算出的旋转矩阵
    RM2 = np.linalg.inv(RM1)  # 求解旋转矩阵的逆
    for i in range(500):
        for j in range(500):
            D = np.dot(RM2, [[i], [j], [1]])  # 旋转后的图像坐标位置 相对应的 原图像坐标位置
            if int(D[0]) >= 500 or int(D[1]) >= 500:  # 旋转后的图像坐标 相对应的 原图像坐标位置 越界
                img2[i, j] = 0
            elif int(D[0]) < 0 or int(D[1]) < 0:  # 旋转后的图像坐标 相对应的 原图像坐标位置 负值
                img2[i, j] = 0
            else:
                img2[i, j] = img1[int(D[0]), int(D[1])]

    # 图像垂直镜像
    RM3 = np.float32([[-1, 0, 500], [0, 1, 0], [0, 0, 1]])  # 计算出的旋转矩阵
    RM4 = np.linalg.inv(RM3)  # 求解旋转矩阵的逆
    for i in range(500):
        for j in range(500):
            D = np.dot(RM4, [[i], [j], [1]])  # 旋转后的图像坐标位置 相对应的 原图像坐标位置
            if int(D[0]) >= 500 or int(D[1]) >= 500:  # 旋转后的图像坐标 相对应的 原图像坐标位置 越界
                img3[i, j] = 0
            elif int(D[0]) < 0 or int(D[1]) < 0:  # 旋转后的图像坐标 相对应的 原图像坐标位置 负值
                img3[i, j] = 0
            else:
                img3[i, j] = img1[int(D[0]), int(D[1])]

    # 图像显示
    fig, axes = plt.subplots(1, 3, figsize=(80, 10), dpi=100)
    axes[0].imshow(img1[:, :, ::-1])
    axes[0].set_title("original")
    axes[1].imshow(img2[:, :, ::-1])
    axes[1].set_title("after horizontal-mirror")
    axes[2].imshow(img3[:, :, ::-1])
    axes[2].set_title("after vertical-mirror")
    plt.show()


def demo2_image():
    # 读取图像
    img1 = cv2.imread("bare.png")
    print(img1.shape)  # (1290, 1080, 3)
    # 图像剪切
    img2 = img1[420:800, 300:800]
    print(img2.shape)  # (380, 500, 3)

    # 图像显示
    fig, axes = plt.subplots(1, 2, figsize=(10, 8), dpi=100)
    axes[0].imshow(img1[:, :, ::-1])
    axes[0].set_title("original")
    axes[1].imshow(img2[:, :, ::-1])
    axes[1].set_title("shear")

    plt.show()


"""
flip(img,flipCode)

img：就是对哪张图片进行翻转

flipCode：flipCode==0 上下翻转  flipCode>0 左右翻转  flipCode<0上下＋左右 翻转
"""


def demo3_image():
    import cv2
    import numpy as np

    ww = cv2.imread("bare.png")
    shangxia = cv2.flip(ww, 0)
    zuoyou = cv2.flip(ww, 1)
    shangxiazuoyou = cv2.flip(ww, -1)

    cv2.imshow('ww', ww)
    cv2.imshow('shangxia', shangxia)
    cv2.imshow('zuoyou', zuoyou)
    cv2.imshow('shangxiazuoyou', shangxiazuoyou)

    cv2.waitKey(0)


"""
图像旋转

"""


def demo4_image():
    ww = cv2.imread("bare.png")

    # 顺时针90
    new = cv2.rotate(ww, cv2.ROTATE_90_CLOCKWISE)
    # 旋转180
    new1 = cv2.rotate(ww, cv2.ROTATE_180)
    # 逆时针旋转90
    new2 = cv2.rotate(ww, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.imshow("ww", ww)
    cv2.imshow('new', new)
    cv2.imshow('new1', new1)
    cv2.imshow('new2', new2)
    cv2.waitKey(0)


if __name__ == '__main__':
    demo4_image()

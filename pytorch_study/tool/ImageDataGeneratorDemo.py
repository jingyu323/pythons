import cv2
from PIL import Image
from keras.src.legacy.preprocessing.image import ImageDataGenerator

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
    image_path = 'bird.png'
    img = Image.open(image_path)
    horizontal_flipped = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    horizontal_flipped.show()
    # cv2.imshow("Image", horizontal_flipped)
    # print("dddd")
    # cv2.waitKey(0)


def image_vertical_flip():
    image_path = 'bird.png'
    img = Image.open(image_path)

    # 垂直翻转
    vertical_flipped = img.transpose(method=Image.FLIP_TOP_BOTTOM)

    # 保存翻转后的图像
    vertical_flipped.save('vertical_flipped_image.png')

    # 显示翻转后的图像 (可选)
    vertical_flipped.show()


if __name__ == '__main__':
    image_vertical_flip()

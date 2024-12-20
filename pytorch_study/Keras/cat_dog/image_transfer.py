import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

import os
import glob

from keras_preprocessing.image import ImageDataGenerator

img = Image.open('./image/superman/001super.png')
im2 = Image.open('./image/superman/002super.png')
im3 = Image.open('./image/superman/003super.png')
# n = 4
# a = np.reshape(np.linspace(0,1,n**2), (n,n))
# plt.figure(figsize=(12, 4.5))
#
# # 第一张图展示灰度的色彩映射方式，并且没有进行颜色的混合
# plt.subplot(131)
# plt.imshow(img, cmap='gray', interpolation='nearest')
# plt.xticks(range(n))
# plt.yticks(range(n))
# # 灰度映射，无混合
# plt.title('Gray color map, no blending', y=1.02, fontsize=12)
#
# # 第二张图展示使用viridis颜色映射的图像，同样没有进行颜色的混合
# plt.subplot(132)
# plt.imshow(im2, cmap='viridis', interpolation='nearest')
# plt.yticks([])
# plt.xticks(range(n))
# # Viridis映射，无混合
# plt.title('Viridis color map, no blending', y=1.02, fontsize=12)
#
# # 第三张图展示使用viridis颜色映射的图像，并且使用了双立方插值方法进行颜色混合
# plt.subplot(133)
# plt.imshow(im3, cmap='viridis', interpolation='bicubic')
# plt.yticks([])
# plt.xticks(range(n))
# # Viridis 映射，双立方混合
# plt.title('Viridis color map, bicubic blending', y=1.02, fontsize=12)
#
# plt.show()

img_path = './image/superman/*'
in_path = './image/'
out_path = './output/'
def print_result(path):
    name_list = glob.glob(path)
    fig = plt.figure(figsize=(10, 6))
    for i in range(3):
        img = Image.open(name_list[i])
        plt.subplot(131 + i)
        plt.imshow(img)
    plt.show()
    print("fig=", fig)
def demo():

    plt.figure(num=1,figsize=(1,1))
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    plt.plot(x, y)
    plt.show()



    name_list = glob.glob(img_path)
    print("name_list=",name_list)
    print_result(img_path)





    datagen = ImageDataGenerator()

    train_generator = datagen.flow_from_directory(
        in_path,
        target_size=(224, 224),  # 图像尺寸
        batch_size=32,  # 批量大小
    )

    print(train_generator.samples)

    gen_data = datagen.flow_from_directory(in_path,
                                           batch_size=1,
                                           shuffle=False,
                                           save_to_dir=out_path + 'resize',
                                           save_prefix='gen',
                                           target_size=(224, 224))
    plt.show()
    print(gen_data.labels)

    print(type(gen_data))
    #返
    # 生成图片
    # for i in range(3):
        # gen_data.next()

    print_result(out_path+'resize/*')


"""
创建一个旋转的数据增强实例，
创建一个数据增强实例，实际上就是直接加载数据
将加载的图像数据重置尺寸
将重置尺寸的图像转换成ndarray格式
将旋转数据增强应用到重置尺寸的图像数据中
使用数据增强生成器重新从目录加载数据
保存加载的数据
使用for循环：
生成并处理三个图像，由于设置了 save_to_dir，这些图像将被保存。
打印三个图像 
"""
def demo3():

    datagen =  ImageDataGenerator(rotation_range=45)
    gen = ImageDataGenerator()
    data = gen.flow_from_directory(in_path, batch_size=1, class_mode=None, shuffle=True, target_size=(224, 224))
    np_data = np.concatenate([data.next() for i in range(data.n)])
    datagen.fit(np_data)
    gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path + 'rotation_range',
                                           save_prefix='gen', target_size=(224, 224))
    for i in range(3):
        gen_data.next()
    print_result(out_path + 'rotation_range/*')

"""
与3中不同的是，这段代码是进行平移变换进行数据增强，指定了平移变换的参数，width_shift_range=0.3，height_shift_range=0.3，
这两个参数分别表示会在水平方向和垂直方向±30%的范围内随机移动
"""
def demo4():
    datagen =  ImageDataGenerator(width_shift_range=0.3, height_shift_range=0.3)
    gen =  ImageDataGenerator()
    data = gen.flow_from_directory(in_path, batch_size=1, class_mode=None, shuffle=True, target_size=(224, 224))
    np_data = np.concatenate([data.next() for i in range(data.n)])
    datagen.fit(np_data)
    gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path + 'shift',
                                           save_prefix='gen', target_size=(224, 224))
    for i in range(3):
        gen_data.next()
    print_result(out_path + 'shift/*')


"""
这里指定缩放参数来进行缩放数据增强
打印结果：
"""
def demo5():
    datagen =  ImageDataGenerator(zoom_range=0.5)
    gen =  ImageDataGenerator()
    data = gen.flow_from_directory(in_path, batch_size=1, class_mode=None, shuffle=True, target_size=(224, 224))
    np_data = np.concatenate([data.next() for i in range(data.n)])
    datagen.fit(np_data)
    gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path + 'zoom',
                                           save_prefix='gen', target_size=(224, 224))
    for i in range(3):
        gen_data.next()
    print_result(out_path + 'zoom/*')


# 这里指定通道偏移参数来进行通道偏移数据增强
def demo6():
    datagen = ImageDataGenerator(channel_shift_range=15)
    gen = ImageDataGenerator()
    data = gen.flow_from_directory(in_path, batch_size=1, class_mode=None, shuffle=True, target_size=(224, 224))
    np_data = np.concatenate([data.next() for i in range(data.n)])
    datagen.fit(np_data)
    gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path + 'channel',
                                           save_prefix='gen', target_size=(224, 224))
    for i in range(3):
        gen_data.next()
    print_result(out_path + 'channel/*')

# 这里指定水平翻转参数来进行水平翻转数据增强
def demo7():
    datagen =  ImageDataGenerator(horizontal_flip=True)
    gen =  ImageDataGenerator()
    data = gen.flow_from_directory(in_path, batch_size=1, class_mode=None, shuffle=True, target_size=(224, 224))
    np_data = np.concatenate([data.next() for i in range(data.n)])
    datagen.fit(np_data)
    gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path + 'horizontal',
                                           save_prefix='gen', target_size=(224, 224))
    for i in range(3):
        gen_data.next()
    print_result(out_path + 'horizontal/*')


# 这里指定rescale重新缩放参数来进行rescale重新缩放数据增强
# 通常用于归一化图像数据。将图像像素值从 [0, 255] 缩放到 [0, 1] 范围，有助于模型的训练
def demo8():
    datagen =  ImageDataGenerator(rescale=1 / 255)
    gen =  ImageDataGenerator()
    data = gen.flow_from_directory(in_path, batch_size=1, class_mode=None, shuffle=True, target_size=(224, 224))
    np_data = np.concatenate([data.next() for i in range(data.n)])
    datagen.fit(np_data)
    gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path + 'rescale',
                                           save_prefix='gen', target_size=(224, 224))
    for i in range(3):
        gen_data.next()
    print_result(out_path + 'rescale/*')

"""
fill_mode='wrap'：当应用几何变换后，图像中可能会出现一些新的空白区域。fill_mode 定义了如何填充这些空白区域。在这种情况下，使用 'wrap' 模式，意味着空白区域将用图像边缘的像素“包裹”填充。
zoom_range=[4, 4]：这设置了图像缩放的范围。在这里，它被设置为在 4 倍范围内进行随机缩放。由于最小和最大缩放因子相同，这将导致所有图像都被放大 4 倍
"""
def demo9():
    datagen =  ImageDataGenerator(fill_mode='wrap', zoom_range=[4, 4])
    gen =  ImageDataGenerator()
    data = gen.flow_from_directory(in_path, batch_size=1, class_mode=None, shuffle=True, target_size=(224, 224))
    np_data = np.concatenate([data.next() for i in range(data.n)])
    datagen.fit(np_data)
    gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path + 'fill_mode',
                                           save_prefix='gen', target_size=(224, 224))
    for i in range(3):
        gen_data.next()
    print_result(out_path + 'fill_mode/*')


"""
‘constant’: kkkkkkkk|abcd|kkkkkkkk (cval=k)
‘nearest’: aaaaaaaa|abcd|dddddddd
‘reflect’: abcddcba|abcd|dcbaabcd
‘wrap’: abcdabcd|abcd|abcdabcd

"""
def demo10():
    datagen =  ImageDataGenerator(fill_mode='constant', zoom_range=[3, 3])
    gen =  ImageDataGenerator()
    data = gen.flow_from_directory(in_path, batch_size=1, class_mode=None, shuffle=True, target_size=(224, 224))
    np_data = np.concatenate([data.next() for i in range(data.n)])
    datagen.fit(np_data)
    gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path + 'fill_mode',
                                           save_prefix='gen', target_size=(224, 224))
    for i in range(3):
        gen_data.next()
    print_result(out_path + 'fill_mode/*')

if __name__ == '__main__':
    demo9()

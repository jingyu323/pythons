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


plt.figure(num=1,figsize=(1,1))
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
plt.plot(x, y)
plt.show()
def print_result(path):
    name_list = glob.glob(path)
    fig = plt.figure(figsize=(10,6))
    for i in range(3):
        img = Image.open(name_list[i])
        plt.subplot(131+i)
        plt.imshow(img)
    plt.show()
    print("fig=",fig)


img_path = './image/superman/*'
in_path = './image/'
out_path = './output/'
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
for i in range(3):
    gen_data.next()

print_result(out_path+'resize/*')





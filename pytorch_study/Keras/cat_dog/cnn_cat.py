import os

import numpy as np
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array

# 数据图像生成
datagen = ImageDataGenerator(
    rotation_range=40,  # 随机旋转度数
    width_shift_range=0.2,  # 随机水平平移
    height_shift_range=0.2,  # 随机竖直平移
    rescale=1 / 255,  # 数据归一化
    shear_range=0.2,  # 随机裁剪
    zoom_range=0.2,  # 随机放大
    horizontal_flip=True,  # 水平翻转
    fill_mode='nearest',  # 填充方式
)

base_dir = 'E:/data/kreas/Kaggle/cat-dog-small/'
train_dir = os.path.join(base_dir, 'train/cats')

# 载入图片
image = load_img(train_dir+'/cat.1.jpg')
x = img_to_array(image)  # 图像数据是一维的，把它转成数组形式
print(x.shape)
x = np.expand_dims(x, 0)  # 在图片的0维增加一个维度，因为Keras处理图片时是4维,第一维代表图片数量
print(x.shape)

#生成20张图片数据
i=0
for batch in datagen.flow(x,batch_size=1,save_to_dir='temp',save_prefix='new_cat',save_format='jpeg'):
    i+=1
    if i==20:
        break
print('finshed!')
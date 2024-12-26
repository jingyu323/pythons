import os

import numpy as np
from keras_core import Sequential
from keras_core.src.layers import Conv2D, Flatten, Dense, Dropout
from keras_core.src.legacy.preprocessing.image import ImageDataGenerator
from keras_core.src.optimizers import Adam
from matplotlib import pyplot as plt
from tf_keras.src.layers import MaxPool2D

# 定义模型
# 将输入数据大小改为150*150*3再加入模型
model = Sequential()
model.add(Conv2D(input_shape=(150, 150, 3), filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

# 卷积池化完原始图像是一个二维的特征图
model.add(Flatten())  # 把二维数据转换为一维
model.add(Dense(64, activation='relu'))  # Dense代表全连接层，64是最后卷积池化后输出的神经元个数
model.add(Dropout(0.5))  # 防止过拟合
model.add(Dense(2, activation='softmax'))  # softmax是把训练结果用概率形式表示的函数，2代表二分类

# 定义优化器
adam = Adam(learning_rate=1e-4)
# 定义优化器、代价函数、训练过程中计算准确率
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# 数据增强
train_datagen = ImageDataGenerator(
    rotation_range=40,  # 随机旋转度数
    width_shift_range=0.2,  # 随机水平平移
    height_shift_range=0.2,  # 随机竖直平移
    rescale=1 / 255,  # 数据归一化
    shear_range=0.2,  # 随机裁剪
    zoom_range=0.2,  # 随机放大
    horizontal_flip=True,  # 水平翻转
    fill_mode='nearest',  # 填充方式
)
test_datagen = ImageDataGenerator(
    rescale=1 / 255,  # 数据归一化
)

batch = 32  # 每次训练传入32张照片
base_dir = 'E:/data/kreas/Kaggle/cat-dog-small/'
train_dir = os.path.join(base_dir, 'train/cats')
test_dir = os.path.join(base_dir, 'test/cats')

# 生成训练数据
train_generator = train_datagen.flow_from_directory(
    train_dir,  # 从训练集这个目录生成数据
    target_size=(150, 150),  # 把生成数据大小定位150*150
    batch_size=batch,
)
# 测试数据
test_generator = test_datagen.flow_from_directory(
    test_dir,  # 从训练集这个目录生成数据
    target_size=(150, 150),  # 把生成数据大小定位150*150
    batch_size=batch,
)
# 查看定义类别分类
print(train_generator.class_indices)

# 定义训练模型
# 传入生成的训练数据、每张图片训练1次，验证数据为生成的测试数据
model.fit_generator(train_generator, epochs=1, validation_data=test_generator)
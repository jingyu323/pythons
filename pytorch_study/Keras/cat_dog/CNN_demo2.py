from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import tensorflow as tf
import keras
import os, shutil

from keras.src.optimizers import RMSprop
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

# 原始目录所在的路径
original_dataset_dir = 'E:/data/kreas/train/train'

# 数据集分类后的目录

base_dir = 'E:/data/kreas/Kaggle/cat-dog-small-test'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# # 训练、验证、测试数据集的目录
train_dir = os.path.join(base_dir, 'train')
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
if not os.path.exists(validation_dir):
    os.makedirs(validation_dir)

test_dir = os.path.join(base_dir, 'test')
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# 猫训练图片所在目录
train_cats_dir = os.path.join(train_dir, 'cats')
if not os.path.exists(train_cats_dir):
    os.makedirs(train_cats_dir)

# 狗训练图片所在目录
train_dogs_dir = os.path.join(train_dir, 'dogs')
if not os.path.exists(train_dogs_dir):
    os.makedirs(train_dogs_dir)

# 猫验证图片所在目录
validation_cats_dir = os.path.join(validation_dir, 'cats')
if not os.path.exists(validation_cats_dir):
    os.makedirs(validation_cats_dir)

# 狗验证数据集所在目录
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
if not os.path.exists(validation_dogs_dir):
    os.makedirs(validation_dogs_dir)

# 猫测试数据集所在目录
test_cats_dir = os.path.join(test_dir, 'cats')
if not os.path.exists(test_cats_dir):
    os.makedirs(test_cats_dir)

# 狗测试数据集所在目录
test_dogs_dir = os.path.join(test_dir, 'dogs')
if not os.path.exists(test_dogs_dir):
    os.makedirs(test_dogs_dir)

# 将前1000张猫图像复制到train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

# 将下500张猫图像复制到validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

# 将下500张猫图像复制到test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

# 将前1000张狗图像复制到train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

# 将下500张狗图像复制到validation_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

# 将下500张狗图像复制到test_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

model = Sequential()
# 卷积层，卷积核是3*3，激活函数relu
model.add(Conv2D(32, (3, 3), activation='relu',
                 input_shape=(150, 150, 3)))
# 最大池化层
model.add(MaxPooling2D((2, 2)))
# 卷积层，卷积核2*2，激活函数relu
model.add(Conv2D(64, (3, 3), activation='relu'))
# 最大池化层
model.add(MaxPooling2D((2, 2)))
# 卷积层，卷积核是3*3，激活函数relu
model.add(Conv2D(128, (3, 3), activation='relu'))
# 最大池化层
model.add(MaxPooling2D((2, 2)))
# 卷积层，卷积核是3*3，激活函数relu
model.add(Conv2D(128, (3, 3), activation='relu'))
# 最大池化层
model.add(MaxPooling2D((2, 2)))
# flatten层，用于将多维的输入一维化，用于卷积层和全连接层的过渡
model.add(Flatten())
# 退出层
model.add(Dropout(0.5))
# 全连接，激活函数relu
model.add(Dense(512, activation='relu'))
# 全连接，激活函数sigmoid
model.add(Dense(1, activation='sigmoid'))
model.summary()

print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=1e-4),
              metrics=['acc'])

# 所有图像将按1/255重新缩放
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    # 这是目标目录
    train_dir,
    # 所有图像将调整为150x150
    target_size=(150, 150),
    batch_size=20,
    # 因为我们使用二元交叉熵损失，我们需要二元标签
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)
# 保存训练得到的的模型
model.save('cats_and_dogs_small_1.keras')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

import os
import shutil

import cv2
import keras
import tensorflow as tf
from keras import Sequential, Input, Model
from keras.api.preprocessing import image
from keras.src import optimizers, models, layers, backend
from keras.src.applications.vgg16 import VGG16, preprocess_input
from keras.src.backend.common.global_state import clear_session

from keras.src.datasets import mnist
from keras.src.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Embedding, LSTM, Conv1D, MaxPooling1D, \
    GlobalAveragePooling1D, SimpleRNN
import numpy as np
from keras.src.legacy.backend import gradients, mean
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.optimizers import SGD, RMSprop
from keras.src.utils import to_categorical
from matplotlib import pyplot as plt
from numpy import gradient
from tensorflow.python.keras.backend import function

from opencv_test.keras_study.common import plot_training


def create_model():
    # 创建一个序列模型
    model = Sequential()
    # 添加一个全连接层，输出空间维度是10（代表10个类别）
    model.add(Dense(units=10, activation='softmax', input_dim=32))
    # 编译模型
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    # 生成随机数据以供模型训练和测试
    data = np.random.random((1000, 32))
    labels = np.random.randint(10, size=(1000, 1))
    # 将数据转换为Keras可用的格式
    labels = keras.utils.to_categorical(labels, num_classes=10)
    # 训练模型
    model.fit(data, labels, epochs=5, batch_size=32)
    # 评估模型
    score = model.evaluate(data, labels)
    # 打印评估得分
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


# 序列模型（Sequential类）
def create_seq_model():
    model = Sequential()
    model.add(Dense(units=10, activation='relu', input_dim=784))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(optimizer='SGD', loss='mse', metrics=['accuracy'])
    x = np.random.random((784, 32))
    y = np.random.randint(10, size=(784, 1))

    model.fit(x, y, batch_size=32, epochs=3, validation_split=0.3)


# 通用模型（Model类）
def create_gen_model():
    x_input = keras.Input(shape=(784,))
    dense_1 = Dense(units=32, activation='relu')(x_input)
    output = Dense(units=10, activation='softmax')(dense_1)
    model = keras.models.Model(inputs=x_input, outputs=output)
    model.compile(optimizer='SGD', loss='mse', metrics=['accuracy'])
    x = np.random.random((784, 32))
    y = np.random.randint(10, size=(784, 1))
    model.fit(x, y, batch_size=32, epochs=3, validation_split=0.3)


def study_demo1():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    data = np.random.random((1000, 100))
    labels = np.random.randint(2, size=(1000, 1))
    # 训练模型，以 32 个样本为一个 batch 进行迭代
    model.fit(data, labels, epochs=10, batch_size=32)


def study_demo2():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=100))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    data = np.random.random((1000, 100))
    labels = np.random.randint(10, size=(1000, 1))
    # 将标签转换为分类的 one-hot 编码
    one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

    # 训练模型，以 32 个样本为一个 batch 进行迭代
    model.fit(data, one_hot_labels, epochs=10, batch_size=32)


# 基于多层感知器 (MLP) 的 softmax 多分类：
def study_mlp():
    x_train = np.random.random((1000, 20))
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
    x_test = np.random.random((100, 20))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

    model = Sequential()
    # Dense(64) 是一个具有 64 个隐藏神经元的全连接层。
    # 在第一层必须指定所期望的输入数据尺寸：
    # 在这里，是一个 20 维的向量。
    model.add(Dense(64, activation='relu', input_dim=20))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    sgd = SGD(learning_rate=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=20,
              batch_size=128)
    score = model.evaluate(x_test, y_test, batch_size=128)
    print(score)


# 基于多层感知器的二分类：
def study_binary_crossent():
    # 生成虚拟数据
    x_train = np.random.random((1000, 20))
    y_train = np.random.randint(2, size=(1000, 1))
    x_test = np.random.random((100, 20))
    y_test = np.random.randint(2, size=(100, 1))

    model = Sequential()
    model.add(Dense(64, input_dim=20, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(x_train, y_train,
                        epochs=20,
                        batch_size=128)
    score = model.evaluate(x_test, y_test, batch_size=128)
    print(score)
    print(history)


def study_vgg_demo():
    # 生成虚拟数据
    x_train = np.random.random((100, 100, 100, 3))
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
    x_test = np.random.random((20, 100, 100, 3))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

    model = Sequential()
    # 输入: 3 通道 100x100 像素图像 -> (100, 100, 3) 张量。
    # 使用 32 个大小为 3x3 的卷积滤波器。
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(learning_rate=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    model.summary()

    model.fit(x_train, y_train, batch_size=32, epochs=10)
    score = model.evaluate(x_test, y_test, batch_size=32)

    print(score)


# 基于 LSTM 的序列分类：
def LSTM_demo():
    max_features = 1024
    x_train = np.random.random((100, 256))
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(256, 1)), num_classes=10)
    x_test = np.random.random((20, 256))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(256, 1)), num_classes=10)

    model = Sequential()
    model.add(Embedding(max_features, output_dim=256))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=16, epochs=10)
    score = model.evaluate(x_test, y_test, batch_size=16)

    print(score)


# 基于 1D 卷积的序列分类：
def Conv1D_demo():
    x_train = np.random.random((100, 256))
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(256, 1)), num_classes=10)
    x_test = np.random.random((20, 256))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(256, 1)), num_classes=10)

    seq_length = 64

    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(seq_length, 100)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=16, epochs=10)
    score = model.evaluate(x_test, y_test, batch_size=16)

    print(score)


def ks_demo():
    # 创建方式2
    model = Sequential()
    # 定义一个卷积输入层，卷积核是3*3，共32个，输入是(28, 28, 1)，输出是(26, 26, 32)
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

    # 定义一个2*2的池化层
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # 将所有的输出展平
    model.add(Flatten())
    # 定义一个全连接层，有64个神经元
    model.add(Dense(64, activation='relu'))
    # 多分类问题，将输出在每个分类上的概率
    model.add(Dense(10, activation='softmax'))
    model.summary()

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    print('train data:', train_images.shape, train_labels.shape)
    print('test data:', test_images.shape, test_labels.shape)

    # 训练数据准确的已经明显优于全连接网络
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5, batch_size=64)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(test_loss, test_acc)


def clean_data():
    original_dataset_dir = 'E://data//kreas//train//train'

    # The directory where we will
    # store our smaller dataset
    base_dir = 'E://data//kreas//Kaggle//cat-dog-small'
    new_create_dir = False
    if not os.path.exists(base_dir):
        new_create_dir = True
        os.mkdir(base_dir)

    # Directories for our training splits
    train_dir = os.path.join(base_dir, 'train')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    train_cats_dir = os.path.join(train_dir, 'cats')
    if not os.path.exists(train_cats_dir):
        os.mkdir(train_cats_dir)
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    if not os.path.exists(train_dogs_dir):
        os.mkdir(train_dogs_dir)

    # Directories for our validation splits
    validation_dir = os.path.join(base_dir, 'validation')
    if not os.path.exists(validation_dir):
        os.mkdir(validation_dir)
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    if not os.path.exists(validation_cats_dir):
        os.mkdir(validation_cats_dir)
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    if not os.path.exists(validation_dogs_dir):
        os.mkdir(validation_dogs_dir)

    # Directories for our test splits
    test_dir = os.path.join(base_dir, 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    test_cats_dir = os.path.join(test_dir, 'cats')
    if not os.path.exists(test_cats_dir):
        os.mkdir(test_cats_dir)
    test_dogs_dir = os.path.join(test_dir, 'dogs')
    if not os.path.exists(test_dogs_dir):
        os.mkdir(test_dogs_dir)

    if new_create_dir or True:
        # Copy first 1000 cat images to train_cats_dir
        fnames = ['cat.{}.jpg'.format(i) for i in range(4000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(train_cats_dir, fname)
            shutil.copyfile(src, dst)

        # Copy next 500 cat images to validation_cats_dir
        fnames = ['cat.{}.jpg'.format(i) for i in range(4000, 8000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(validation_cats_dir, fname)
            shutil.copyfile(src, dst)

        # Copy next 500 cat images to test_cats_dir
        fnames = ['cat.{}.jpg'.format(i) for i in range(8000, 8500)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(test_cats_dir, fname)
            shutil.copyfile(src, dst)

        # Copy first 1000 dog images to train_dogs_dir
        fnames = ['dog.{}.jpg'.format(i) for i in range(4000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(train_dogs_dir, fname)
            shutil.copyfile(src, dst)

        # Copy next 500 dog images to validation_dogs_dir
        fnames = ['dog.{}.jpg'.format(i) for i in range(4000, 8000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(validation_dogs_dir, fname)
            shutil.copyfile(src, dst)

        # Copy next 500 dog images to test_dogs_dir
        fnames = ['dog.{}.jpg'.format(i) for i in range(8000, 8500)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(test_dogs_dir, fname)
            shutil.copyfile(src, dst)

    return train_dir, validation_dir, test_dir


def CNN_keras_demo():
    train_dir, validation_dir, test_dir = clean_data()

    # All images will be rescaled by 1./255
    # train_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # 通过对ImageDataGenerator实例读取的图像执行多次随机变换不断的丰富训练样本
    train_augmented_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,  # 随机旋转的角度范围
        width_shift_range=0.2,  # 在水平方向上平移的范围
        height_shift_range=0.2,  # 在垂直方向上平移的范围
        shear_range=0.2,  # 随机错切变换的角度
        zoom_range=0.2,  # 随机缩放的范围
        horizontal_flip=True, )  # 随机将一半图像水平翻转

    # Note that the validation data should not be augmented!
    # train_augmented_generator = train_augmented_datagen.flow_from_directory(
    #     train_dir,
    #     target_size=(150, 150),
    #     batch_size=32,
    #     class_mode='binary')

    # 分批次的将数据按目录读取出来，ImageDataGenerator会一直取图片，直到break
    train_generator = train_augmented_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    print("========================")

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
    # 四卷积层、四MaxPooling、一展开层、一全连接层、一输出层的基准网络
    model1 = Sequential()
    model1.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model1.add(MaxPooling2D((2, 2)))
    model1.add(Conv2D(64, (3, 3), activation='relu'))
    model1.add(MaxPooling2D((2, 2)))
    model1.add(Conv2D(128, (3, 3), activation='relu'))
    model1.add(MaxPooling2D((2, 2)))
    model1.add(Conv2D(128, (3, 3), activation='relu'))
    model1.add(MaxPooling2D((2, 2)))
    model1.add(Flatten())
    model1.add(Dense(512, activation='relu'))
    model1.add(Dense(1, activation='sigmoid'))
    model1.summary()

    model1.compile(loss='binary_crossentropy',
                   optimizer=optimizers.RMSprop(learning_rate=1e-4),
                   metrics=['acc'])

    history = model1.fit(
        train_generator,  # 训练数据生成器
        steps_per_epoch=100,  # 每一个迭代需要读取100次生成器的数据
        epochs=30,  # 迭代次数
        validation_data=validation_generator,  # 验证数据生成器

        validation_steps=180)  # 需要读取50次才能加载全部的验证集数据
    # validation_steps 需要设置为epochs 的整数倍才行
    # loss的波动幅度有点大
    print(model1.metrics_names)
    print(model1.evaluate(test_generator, steps=50))

    accuracy = history.history["acc"]
    val_accuracy = history.history["val_acc"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(accuracy) + 1)
    plt.plot(epochs, accuracy, "bo", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()


'''
增加图片反转 
增加随机丢失层

'''


def CNN_keras_demo2():
    print("CNN_keras_demo2 .... ")
    train_dir, validation_dir, test_dir = clean_data()
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_augmented_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,  # 随机旋转的角度范围
        width_shift_range=0.2,  # 在水平方向上平移的范围
        height_shift_range=0.2,  # 在垂直方向上平移的范围
        shear_range=0.2,  # 随机错切变换的角度
        zoom_range=0.2,  # 随机缩放的范围
        horizontal_flip=True, )  # 随机将一半图像水平翻转

    # Note that the validation data should not be augmented!
    train_augmented_generator = train_augmented_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    print("========================")

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    # 重新训练一个模型
    model2 = Sequential()
    model2.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model2.add(MaxPooling2D((2, 2)))
    model2.add(Conv2D(64, (3, 3), activation='relu'))
    model2.add(MaxPooling2D((2, 2)))
    model2.add(Conv2D(128, (3, 3), activation='relu'))
    model2.add(MaxPooling2D((2, 2)))
    model2.add(Conv2D(128, (3, 3), activation='relu'))
    model2.add(MaxPooling2D((2, 2)))
    model2.add(Flatten())
    model2.add(Dropout(0.5))  # 新加了dropout层
    model2.add(Dense(512, activation='relu'))
    model2.add(Dense(1, activation='sigmoid'))
    model2.summary()
    model2.compile(loss='binary_crossentropy',
                   optimizer=optimizers.RMSprop(learning_rate=1e-4),
                   metrics=['acc'])

    history2 = model2.fit(
        train_augmented_generator,
        steps_per_epoch=100,  # 每一批次读取100轮数据，总共是3200张图片
        epochs=50,
        validation_data=validation_generator,
        validation_steps=50)

    # loss的波动幅度有点大
    print(model2.metrics_names)
    print(model2.evaluate(test_generator, steps=50))


def CNN_keras_VGG16():
    train_dir, validation_dir, test_dir = clean_data()
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_augmented_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,  # 随机旋转的角度范围
        width_shift_range=0.2,  # 在水平方向上平移的范围
        height_shift_range=0.2,  # 在垂直方向上平移的范围
        shear_range=0.2,  # 随机错切变换的角度
        zoom_range=0.2,  # 随机缩放的范围
        horizontal_flip=True, )  # 随机将一半图像水平翻转

    # Note that the validation data should not be augmented!
    train_augmented_generator = train_augmented_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    print("========================")

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    from keras.src.applications.vgg16 import VGG16
    conv_base = VGG16(weights='imagenet',  # 指定模型初始化的权重检查点
                      include_top=False,  # 模型最后是否包含密集连接分类器，默认有1000个类别
                      input_shape=(150, 150, 3))
    conv_base.trainable = False
    conv_base.summary()

    model4 = Sequential()
    # model4.add(Flatten(input_shape=conv_base.output_shape[1:]))

    model4.add(conv_base.input)
    model4.add(Flatten())
    model4.add(Dense(256, activation='relu'))
    model4.add(Dense(1, activation='sigmoid'))
    model4.summary()

    model = Model(
        inputs=conv_base.input,
        outputs=model4(conv_base.output))
    model.summary()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(learning_rate=0.0001),
        metrics=['accuracy'])

    #
    #
    #
    # print('This is the number of trainable weights '
    #       'before freezing the conv base:', len(model4.trainable_weights))
    #
    #
    #
    # print('This is the number of trainable weights '
    #       'before freezing the conv base:', len(model4.trainable_weights))
    #
    #
    model4.compile(loss='binary_crossentropy',
                   optimizer=optimizers.RMSprop(learning_rate=2e-5),
                   metrics=['acc'])

    history4 = model4.fit(
        train_augmented_generator,
        steps_per_epoch=100,  # 3200个输入图片，增强
        epochs=30,
        validation_data=validation_generator,
        validation_steps=30,
        verbose=2)

    print(model4.metrics_names)
    print(model4.evaluate(test_generator, steps=50))


def CNN_keras_VGG16_demo():
    # conv_base = VGG16(include_top=False,
    #                   weights='imagenet',
    #                   input_shape=(150, 150, 3))

    conv_base = keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(150, 150, 3),
        pooling=None,
        classifier_activation="softmax",
    )

    conv_base.trainable = False  # 冻结参数，使之不被更新
    conv_base.summary()

    model = Sequential()

    print(conv_base.input)
    model.add(conv_base.input)
    model.add(layers.Flatten())

    model.add(Dropout(0.5))  # 新加了dropout层
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    train_dir, validation_dir, test_dir = clean_data()
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_augmented_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,  # 随机旋转的角度范围
        width_shift_range=0.2,  # 在水平方向上平移的范围
        height_shift_range=0.2,  # 在垂直方向上平移的范围
        shear_range=0.2,  # 随机错切变换的角度
        zoom_range=0.2,  # 随机缩放的范围
        horizontal_flip=True)  # 随机将一半图像水平翻转

    # Note that the validation data should not be augmented!
    train_augmented_generator = train_augmented_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    print("========================")

    conv_base.trainable = True

    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    #
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(learning_rate=2e-5),
                  metrics=['acc'])

    history4 = model.fit(
        train_augmented_generator,
        steps_per_epoch=100,  # 3200个输入图片，增强
        epochs=60,
        validation_data=validation_generator,
        validation_steps=30)

    plot_training(history4)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    print(model.metrics_names)
    print(model.evaluate(test_generator, steps=50))

def g(x1, x2):
    return (x1 + 5) * (x2 ** 2)

def vgg_vison():
    keras.api.backend.clear_session()
    model = VGG16(weights='imagenet')
    img_path = '../image/cat.jpg'

    # `img` is a PIL image of size 224x224
    img = image.load_img(img_path, target_size=(224, 224))

    # `x` is a float32 Numpy array of shape (224, 224, 3)
    x = image.img_to_array(img)

    # We add a dimension to transform our array into a "batch"
    # of size (1, 224, 224, 3)
    x = np.expand_dims(x, axis=0)

    # 将进行颜色标准化
    x = preprocess_input(x)

    # 预测，并打印TOP3的分类
    preds = model.predict(x)

    # This is the "african elephant" entry in the prediction vector
    african_elephant_output = model.output[:, 386]
    # The is the output feature map of the `block5_conv3` layer,
    # the last convolutional layer in VGG16
    last_conv_layer = model.get_layer('block5_conv3')

    # This is the gradient of the "african elephant" class with regard to
    # the output feature map of `block5_conv3`





    grads =   gradient(african_elephant_output, last_conv_layer.output)[0]

    # This is a vector of shape (512,), where each entry
    # is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = mean(grads, axis=(0, 1, 2))

    # This function allows us to access the values of the quantities we just defined:
    # `pooled_grads` and the output feature map of `block5_conv3`,
    # given a sample image
    iterate =  function([model.input], [pooled_grads, last_conv_layer.output[0]])

    # These are the values of these two quantities, as Numpy arrays,
    # given our sample image of two elephants
    pooled_grads_value, conv_layer_output_value = iterate([x])

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the elephant class
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # 小于0则设成0
    heatmap /= np.max(heatmap)  # 除最大值

    img = cv2.imread(img_path)

    # We resize the heatmap to have the same size as the original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # We convert the heatmap to RGB
    heatmap = np.uint8(255 * heatmap)

    # We apply the heatmap to the original image
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 0.4 here is a heatmap intensity factor
    superimposed_img = heatmap * 0.4 + img
    cv2.imshow('superimposed', superimposed_img)


def keras_demo11():


    samples = ['The cat sat on the mat.', 'The dog ate my homework.']

    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(samples)

    one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
    print('texts_to_matrix')
    print(one_hot_results.shape)
    print(one_hot_results)

# 循环神经网络  会将一个状态传递到下次
# 循环神经网络（RNN、recurrent neural network）区别于传统的网络结构，增加了一个状态（state），每次处理的时候输入为本次输入+当前状态

def keras_demo_RNN_01():

    model = Sequential()
    model.add(Embedding(10000, 32))
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(SimpleRNN(32))  # This last layer only returns the last outputs.
    model.summary()

'''
随着层数的增加容易出现梯度消失，增加网络层数将变得无法训练，继而就有了长短期记忆（LSTM，long short-term memory)
LSTM增加了一种携带信息跨越多个时间步的方法 —— Ct
'''

def keras_demo_LSTM_01():
    max_features =100
    float_data=[]
    input_train=[]
    y_train=[]
    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))
    '''
    门控循环单元（GRU，gated recurrent unit）层的工作原理与 LSTM相同，但它做了一些简化，运行的计算代价更低，效果可能不如LSTM
    '''
    model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    history = model.fit(input_train, y_train,
                        epochs=10,
                        batch_size=128,
                        validation_split=0.2)


def keras_demo_rnn_02():
    float_data=[]
    model = Sequential()
    model.add(layers.GRU(32,
                         dropout=0.1,
                         recurrent_dropout=0.5,
                         return_sequences=True,
                         input_shape=(None, float_data.shape[-1])))
    # 堆叠➕一层
    model.add(layers.GRU(64, activation='relu',
                         dropout=0.1,
                         recurrent_dropout=0.5))
    model.add(layers.Dense(1))

    model.compile(optimizer=RMSprop(), loss='mae')
    model.summary()


def keras_demo_rnn_03():
    seq_model = Sequential()
    seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
    seq_model.add(layers.Dense(32, activation='relu'))
    seq_model.add(layers.Dense(10, activation='softmax'))
    # seq_model.summary()

    seq_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    x_train = np.random.random((1000, 64))
    y_train = np.random.random((1000, 10))
    seq_model.fit(x_train, y_train, epochs=10, batch_size=128)
    score = seq_model.evaluate(x_train, y_train)
    print(score)



# 函数式 写法
def keras_demo_rnn_04():
    input_tensor = Input(shape=(64,))
    x = layers.Dense(32, activation='relu')(input_tensor)
    x = layers.Dense(32, activation='relu')(x)
    output_tensor = layers.Dense(10, activation='softmax')(x)
    model = Model(input_tensor, output_tensor)
    # model.summary()

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    x_train = np.random.random((1000, 64))
    y_train = np.random.random((1000, 10))
    model.fit(x_train, y_train, epochs=10, batch_size=128)
    score = model.evaluate(x_train, y_train)

    print(score)

def keras_demo_rnn_05():
    text_vocabulary_size = 10000
    question_vocabulary_size = 10000
    answer_vocabulary_size = 500

    text_input = Input(shape=(None,), dtype='int32', name='text')
    embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)
    encoded_text = layers.LSTM(32)(embedded_text)
    question_input = Input(shape=(None,), dtype='int32', name='question')
    embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)
    encoded_question = layers.LSTM(16)(embedded_question)

    concatenated = layers.concatenate([encoded_text, encoded_question],
                                      axis=-1)
    answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)

    model = Model([text_input, question_input], answer)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    num_samples = 1000
    max_length = 100
    text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))
    question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))
    answers = np.random.randint(answer_vocabulary_size, size=(num_samples))
    answers = keras.utils.to_categorical(answers, answer_vocabulary_size)

    # 方法一
    model.fit([text, question], answers, epochs=10, batch_size=128)
    # 方法二，对应输入名字的字典形式
    # model.fit({'text': text, 'question': question}, answers, epochs=10, batch_size=128)


def  keras_demo_rnn_06():
    vocabulary_size = 50000
    num_income_groups = 10

    posts_input = Input(shape=(None,), dtype='int32', name='posts')
    embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)
    x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
    x = layers.MaxPooling1D(5)(x)
    x = layers.Conv1D(256, 5, activation='relu')(x)
    x = layers.Conv1D(256, 5, activation='relu')(x)
    x = layers.MaxPooling1D(5)(x)
    x = layers.Conv1D(256, 5, activation='relu')(x)
    x = layers.Conv1D(256, 5, activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)

    age_prediction = layers.Dense(1, name='age')(x)
    income_prediction = layers.Dense(num_income_groups, activation='softmax', name='income')(x)
    gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)
    model = Model(posts_input, [age_prediction, income_prediction, gender_prediction])

    # 编译，方法一，数组格式
    model.compile(optimizer='rmsprop',
                  loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'])
    # 方法二，如果有名字的话可以用字典的方式
    model.compile(optimizer='rmsprop',
                  loss={'age': 'mse',
                        'income': 'categorical_crossentropy',
                        'gender': 'binary_crossentropy'})
    # 方法三，增加权重
    model.compile(optimizer='rmsprop',
                  loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'],
                  loss_weights=[0.25, 1., 10.])
    # 方法四，增加权重
    model.compile(optimizer='rmsprop',
                  loss={'age': 'mse',
                        'income': 'categorical_crossentropy',
                        'gender': 'binary_crossentropy'},
                  loss_weights={'age': 0.25,
                                'income': 1.,
                                'gender': 10.})
    # posts=[]
    # # 训练，方法一，
    # model.fit(posts, [age_targets, income_targets, gender_targets],
    #           epochs=10,
    #           batch_size=64)
    # # 方法二
    # model.fit(posts, {'age': age_targets,
    #                   'income': income_targets,
    #                   'gender': gender_targets},
    #           epochs=10,
    #           batch_size=64)



if __name__ == '__main__':
    # create_seq_model()
    # LSTM_demo()
    # CNN_keras_demo()
    # CNN_keras_VGG16()
    # 精度不行不到60
    # CNN_keras_VGG16_demo()
    # vgg_vison()
    # keras_demo_RNN_01()
    # keras_demo_rnn_03()
    keras_demo_rnn_05()

import os
import shutil

import keras
from keras import Sequential, Input, Model
from keras.src import optimizers, models, layers
from keras.src.applications.vgg16 import VGG16

from keras.src.datasets import mnist
from keras.src.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Embedding, LSTM, Conv1D, MaxPooling1D, \
    GlobalAveragePooling1D
import numpy as np
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.optimizers import SGD
from keras.src.utils import to_categorical
from matplotlib import pyplot as plt

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

    model.fit(x,y,batch_size=32, epochs=3, validation_split=0.3)

# 通用模型（Model类）
def create_gen_model():
    x_input = keras.Input(shape=(784,))
    dense_1 = Dense(units=32, activation='relu')(x_input)
    output = Dense(units=10, activation='softmax')(dense_1)
    model = keras.models.Model(inputs=x_input, outputs=output)
    model.compile(optimizer='SGD', loss='mse', metrics=['accuracy'])
    x = np.random.random((784, 32))
    y = np.random.randint(10, size=(784, 1))
    model.fit(x,y,batch_size=32, epochs=3, validation_split=0.3)


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
    print( score)
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

    history= model.fit(x_train, y_train,
              epochs=20,
              batch_size=128)
    score = model.evaluate(x_test, y_test, batch_size=128)
    print( score)
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
def  Conv1D_demo():
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
    model =  Sequential()
    # 定义一个卷积输入层，卷积核是3*3，共32个，输入是(28, 28, 1)，输出是(26, 26, 32)
    model.add( Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

    # 定义一个2*2的池化层
    model.add( MaxPooling2D((2, 2)))
    model.add( Conv2D(64, (3, 3), activation='relu'))
    model.add( MaxPooling2D((2, 2)))
    model.add( Conv2D(64, (3, 3), activation='relu'))
    # 将所有的输出展平
    model.add( Flatten())
    # 定义一个全连接层，有64个神经元
    model.add( Dense(64, activation='relu'))
    # 多分类问题，将输出在每个分类上的概率
    model.add( Dense(10, activation='softmax'))
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


def  clean_data():
    original_dataset_dir = 'E://data//kreas//train//train'

    # The directory where we will
    # store our smaller dataset
    base_dir = 'E://data//kreas//Kaggle//cat-dog-small'
    new_create_dir=False
    if  not  os.path.exists(base_dir):
        new_create_dir= True
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


    if new_create_dir or  True:
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


    return  train_dir,validation_dir,test_dir


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
    model2 =  Sequential()
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
    model4.add( Flatten())
    model4.add( Dense(256, activation='relu'))
    model4.add( Dense(1, activation='sigmoid'))
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
    model.add( layers.Flatten())
    model.add( Dense(256, activation='relu'))
    model.add( Dense(1, activation='sigmoid'))
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
        validation_steps=30 )

    plot_training(history4)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')


    print(model.metrics_names)
    print(model.evaluate(test_generator, steps=50))





if __name__ == '__main__':
    # create_seq_model()
    # LSTM_demo()
    # CNN_keras_demo()
    # CNN_keras_VGG16()
    CNN_keras_VGG16_demo()
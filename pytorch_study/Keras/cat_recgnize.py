import os

from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.legacy.preprocessing.image import ImageDataGenerator

from keras.src.optimizers import Adam


def  demo_cnn():
    model = Sequential()
    model.add(Conv2D(input_shape=(150, 150, 3), filters=32, kernel_size=3, strides=1, padding="same",
                            activation="relu"))
    model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding="valid"))

    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding="valid"))

    model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding="valid"))

    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation="softmax"))

    # 定义优化器
    adam = Adam(learning_rate=1e-4)

    # 定义优化器，loss_function,训练过程中计算准确率
    model.compile(
        optimizer=adam,
        loss="categorical_crossentropy",
        metrics=['accuracy']
    )

    # 查看模型的结构
    model.summary()

    # 训练集数据生成
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,  # 归一化处理
        shear_range=0.2,  # 随机裁剪
        zoom_range=0.2,  # 图片放大
        horizontal_flip=True  # 水平翻转
    )
    # 测试集数据处理
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    batch_size = 32
    # 生成训练数据
    train_generator = train_datagen.flow_from_directory(
        "E:/data/kreas/train/train",  # 训练数据路径
        target_size=(150, 150),  # 设置图片大小
        batch_size=batch_size  # 批次大小
    )
    # 测试数据
    test_generator = test_datagen.flow_from_directory(
        "E:/data/kreas/test1",  # 训练数据路径
        target_size=(150, 150),  # 设置图片大小
        batch_size=batch_size  # 批次大小
    )

    # 统计文件个数
    totalFileCount_train = sum(
        [len(files) for root, dirs, files in os.walk("E:/data/kreas/train/train")])
    totalFileCount_test = sum([len(files) for root, dirs, files in os.walk("E:/data/kreas/test1")])
    model.fit(
        x=train_generator,
        steps_per_epoch=totalFileCount_train / batch_size,
        epochs=50,
        validation_data=test_generator,
        validation_steps=1000 / batch_size
    )



    # 保存模型
    # model.save("CNN1.h5")




if __name__ == '__main__':

    demo_cnn();
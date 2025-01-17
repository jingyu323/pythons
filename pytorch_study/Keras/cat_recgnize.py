import os

from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.legacy.preprocessing.image import ImageDataGenerator

from keras.src.optimizers import Adam
import matplotlib.pyplot as plt
from keras.src.saving import load_model


def demo_cnn():
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
    # 进行分类
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
        "E:/data/kreas/Kaggle/cat-dog-small/train",  # 训练数据路径
        target_size=(150, 150),  # 设置图片大小
        batch_size=batch_size  # 批次大小
    )
    # 测试数据
    test_generator = test_datagen.flow_from_directory(
        "E:/data/kreas/Kaggle/cat-dog-small/test",  # 训练数据路径
        target_size=(150, 150),  # 设置图片大小
        batch_size=batch_size  # 批次大小
    )

    # 统计文件个数
    totalFileCount_train = sum(
        [len(files) for root, dirs, files in os.walk("E:/data/kreas/Kaggle/cat-dog-small/train")])
    totalFileCount_test = sum([len(files) for root, dirs, files in os.walk("E:/data/kreas/Kaggle/cat-dog-small/test")])
    if os.path.exists("CNN1.keras"):
        model = load_model('CNN1.keras')
        print("CNN1.keras  exists")

    else:

        history = model.fit(
            train_generator,
            steps_per_epoch=int(totalFileCount_train / batch_size),
            epochs=50,
            validation_data=test_generator,
            validation_steps=int(1000 / batch_size)
        )

        # 保存模型
        model.save("CNN1.keras")

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

    plt.figure(figsize=(18, 3))
    plt.subtitle("预测结果显示")

    # predictions = model.predict(img_array)
    # for images, labels in val_ds.take(1):
    #     for i in range(8):
    #         ax = plt.subplot(2, 4, i + 1)
    #
    #         # 显示图片
    #         ax.imshow(images[i].numpy())
    #
    #         # 需要给图片增加一个维度
    #         img_array = tf.expand_dims(images[i], 0)
    #
    #         # 使用模型预测图片中的人物
    #         predictions = model.predict(img_array)
    #         plt.title(class_names[np.argmax(predictions)])
    #
    #         plt.axis("off")


if __name__ == '__main__':
    demo_cnn();

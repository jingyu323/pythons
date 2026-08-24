# 导入所需模块
from imutils.paths import list_images
from keras import backend as K, Sequential
from keras.src.initializers import TruncatedNormal
from keras.src.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
from keras.src.optimizers import SGD
from keras.src.utils import plot_model
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


class SimpleVGGNet:
    @staticmethod
    def build(width, height, depth, classes):  # 长 宽 深度（特征图的个数）
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # CONV => RELU => POOL
        model.add(Conv2D(32, (3, 3), padding="same",
                         input_shape=inputShape, kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        #  model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #  model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        #  model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        #  model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #  model.add(Dropout(0.25))

        # (CONV => RELU) * 3 => POOL
        model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        #  model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        #  model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        #  model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #  model.add(Dropout(0.25))

        # FC层
        model.add(Flatten())
        model.add(Dense(256, kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        #  model.add(BatchNormalization())
        #  model.add(Dropout(0.6))

        # softmax 分类
        model.add(Dense(classes, kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("softmax"))

        return model



import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import keras

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 读取数据和标签
print("------开始读取数据------")
data = []
labels = []

# 拿到图像数据路径，方便后续读取
imagePaths = sorted(list(list_images('E:/data/kreas/train/tmp')))
random.seed(42)
random.shuffle(imagePaths)

# 遍历读取数据
for imagePath in imagePaths:
    # 读取图像数据
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (64, 64))  # 将图片resize为相同尺寸
    data.append(image)
    # 读取标签
    label = imagePath.split(os.path.sep)[-2]  # 根据文件夹获取标签
    labels.append(label)

# 对图像数据做scale操作
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# 数据集切分
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# 转换标签为one-hot encoding格式（三分类及以上需要，二分类不需要）
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# 数据增强处理
"""
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")
"""

# 建立卷积神经网络
model = SimpleVGGNet.build(width=64, height=64, depth=3, classes=len(lb.classes_))

# 设置初始化超参数
INIT_LR = 0.01
EPOCHS = 30
BS = 32

# 损失函数，编译模型
print("------准备训练网络------")
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=INIT_LR,
    decay_steps=10,
    decay_rate=0.98)
opt = SGD(learning_rate=lr_schedule)  # 一开始的权重参数较好，可以把学习参数设置的较大，后续权重参数变差，学习参数也设置较低
# one-hot编码用loss="CategoricalCrossentropy" 数组编码用loss="SparseCategoricalCrossentropy"
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# plot_model(model)
# 训练网络模型
"""
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS)
"""

H = model.fit(trainX, trainY, validation_data=(testX, testY),
              epochs=EPOCHS, batch_size=32)

# 测试
print("------测试网络------")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=lb.classes_))

# 绘制结果曲线
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('./output_cnn/cnn_plot.png')

# 保存模型
print("------正在保存模型------")
model.save('./output_cnn/cnn.model')
f = open('./output_cnn/cnn_lb.pickle', "wb")
f.write(pickle.dumps(lb))
f.close()
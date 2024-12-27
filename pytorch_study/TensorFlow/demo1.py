# 1. 准备数据

import numpy as np
 

import tensorflow as tf
import os

from keras import Sequential, models
from keras.src.layers import Dense
 
import tensorflow as tf
from keras import Sequential 

from tensorflow.python.keras import models 


def demo():
    x_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    y_train = np.array([[0], [0], [1], [1]])  # 标签为0或1

    # 2. 定义模型
    model = Sequential([
        Dense(1, activation='sigmoid', input_shape=(2,))
    ])

    # 3. 编译模型
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

    # 4. 训练模型
    model.fit(x_train, y_train, epochs=100, verbose=1)

    # 5. 预测
    predictions = model.predict(x_train)
    print(predictions)

def demo1():
    # 加载MNIST数据集
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 数据预处理
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # 构建模型
    model = models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    # 编译模型
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # 训练模型
    model.fit(x_train, y_train, epochs=5)

    # 评估模型
    model.evaluate(x_test, y_test)


if __name__ == '__main__':


    print(tf.__version__)

    demo1()

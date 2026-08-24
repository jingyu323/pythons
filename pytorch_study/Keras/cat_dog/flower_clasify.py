import keras
import numpy as np
from keras import Sequential, Input
from keras.src.layers import Dense
from keras.src.optimizers import SGD
from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Flatten,Dense
from tensorflow.keras import Model
"""第二步：指定输入网络的训练集和测试集"""
def demo():


    x_train = datasets.load_iris().data
    y_train = datasets.load_iris().target


    print(x_train)
    """第三步：逐层搭建神经网络结构"""
    model = Sequential([Dense(3, activation='softmax' , kernel_regularizer=keras.regularizers.l2())
    ])
    """第四步：在model.compile()中配置训练方法"""
    model.compile(optimizer= SGD(learning_rate=0.1),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])
    """第五步：在model.fit()中训练模型"""
    print("x_train:",type(x_train))
    print("x_train:",x_train.shape)
    print("y_train:",type(y_train))
    print("y_train:",y_train.shape)


    model.summary()
    model.fit(x_train, y_train, batch_size=64, epochs=50, validation_split=0.2 )



    """第六步：使用model.summary()打印网络结构，统计网络参数"""

    """预测一下吧"""
    x_new = np.array([[2, 3, 4, 1]])
    y_pred = model.predict(x_new)
    print(np.argmax(y_pred, axis=1))  # 输出类别


def demo2():
    import tensorflow as tf
    """第二部：准备所用数据集"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train / 250, x_test / 250  # 处理一下特征，便于收敛
    """第三步：逐层搭建网络结构"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='sigmoid'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    """第四步：在model.compile()中配置训练参数"""
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    """第五步：在model.fit()中训练网络"""
    model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test), validation_freq=2)
    """第六步：使用model.summary()打印网络结构"""
    model.summary()

    res=model.predict(x_test)
    print(res)

def demo3():
    """第二步：准备数据集"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train / 300, x_test / 300  # 将特征处理一下，使其位于[0,1],有助于收敛
    """第三步：使用class类搭建网络结构"""

    class MnistModel(Model):
        def __init__(self):
            super(MnistModel, self).__init__()
            self.flatten = Flatten()
            self.d1 = Dense(128, activation='relu')
            self.d2 = Dense(10, activation='softmax')

        def call(self, x):
            x = self.flatten(x)
            x = self.d1(x)
            y = self.d2(x)
            return y

    model = MnistModel()
    """第四步：使用model.compile()配置网络训练参数"""
    model.compile(optimizer='sgd',  # 优化器
                  loss='sparse_categorical_crossentropy',  # 损失函数
                  metrics=['sparse_categorical_accuracy'])  # 准确率评级标准
    """第五步：使用model.fit()训练网络"""
    model.fit(x_train, y_train, batch_size=88, epochs=30, validation_data=(x_test, y_test), validation_freq=4)
    """第六步：使用model.summary打印网络结构"""
    model.summary()
    res=model.predict(x_test)
    print(res)
if __name__ == '__main__':

    demo3()
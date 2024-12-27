import keras
import numpy as np
from keras import Sequential, Input
from keras.src.layers import Dense
from keras.src.optimizers import SGD
from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf

"""第二步：指定输入网络的训练集和测试集"""

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


import keras
import numpy
from keras import Sequential
from keras.src.datasets import mnist
from keras.src.layers import Dense, Activation, Dropout


import numpy as np
from keras.src.optimizers import SGD

print(keras.version())

# 生成随机数据
X = np.random.rand(1000, 20)  # 1000个样本，20个特征
y = np.random.randint(0, 2, (1000, 1))  # 二分类目标

model = Sequential()

'''
第二步：构建网络层
'''
model.add(Dense(500, input_shape=(784,)))  # 输入层，28*28=784
model.add(Activation('tanh'))  # 激活函数是tanh
model.add(Dropout(0.5))  # 采用50%的dropout

model.add(Dense(500))  # 隐藏层节点500个
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(10))  # 输出结果是10个类别，所以维度是10
model.add(Activation('softmax'))  # 最后一层用softmax作为激活函数

'''
第三步：编译
'''

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)  # 优化函数，设定学习率（lr）等参数

model.compile(loss='categorical_crossentropy', optimizer=sgd )  # 使用交叉熵作为loss函数

'''
第四步：训练
.fit的一些参数
batch_size：对总的样本数进行分组，每组包含的样本数量
epochs ：训练次数
shuffle：是否把数据随机打乱之后再进行训练
validation_split：拿出百分之多少用来做交叉验证
verbose：屏显模式 0：不输出 1：输出进度 2：输出每次的训练结果
'''
(X_train, y_train), (X_test, y_test) = mnist.load_data()  # 使用Keras自带的mnist工具读取数据（第一次需要联网）
# 由于mist的输入数据维度是(num, 28, 28)，这里需要把后面的维度直接拼起来变成784维
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
Y_train = (numpy.arange(10) == y_train[:, None]).astype(int)
Y_test = (numpy.arange(10) == y_test[:, None]).astype(int)

model.fit(X_train, Y_train, batch_size=200, epochs=50, shuffle=True, verbose=1, validation_split=0.3)
model.evaluate(X_test, Y_test, batch_size=200, verbose=1)

'''
第五步：输出
'''
print("test set")
scores = model.evaluate(X_test, Y_test, batch_size=200, verbose=0)
print("")
print("The test loss is %f" % scores)
result = model.predict(X_test, batch_size=200, verbose=0)

result_max = numpy.argmax(result, axis=1)
test_max = numpy.argmax(Y_test, axis=1)

result_bool = numpy.equal(result_max, test_max)
true_num = numpy.sum(result_bool)
print("")
print("The accuracy of the model is %f" % (true_num / len(result_bool)))

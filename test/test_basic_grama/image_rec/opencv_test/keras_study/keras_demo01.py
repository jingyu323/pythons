import keras
from keras import Sequential
from keras.src.layers import Dense, Dropout
import numpy as np
from keras.src.optimizers import SGD


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
    x_input = keras.layers.Input(shape=(784,))
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

if __name__ == '__main__':
    # create_seq_model()
    study_mlp()
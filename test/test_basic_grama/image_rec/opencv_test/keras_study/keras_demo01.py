import keras
from keras import Sequential, Input, Model
from keras.src import optimizers
from keras.src.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Embedding, LSTM, Conv1D, MaxPooling1D, \
    GlobalAveragePooling1D
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
    # 创建方式1
    # model =  Sequential()
    # model.add( Dense(32, activation='relu', input_shape=(784,)))
    # model.add(Dense(10, activation='softmax'))
    # model.compile(optimizer=optimizers.RMSprop(lr=0.001),
    #               loss='mse',
    #               metrics=['accuracy'])
    # model.fit(input_tensor, target_tensor, batch_size=128, epochs=10)

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




if __name__ == '__main__':
    # create_seq_model()
    # LSTM_demo()
    ks_demo()
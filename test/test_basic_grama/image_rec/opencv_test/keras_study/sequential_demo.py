import keras
import numpy as np
from keras import Sequential, Input, Model
from keras.src.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Embedding, LSTM
from keras.src.optimizers import SGD


# 对于具有 2 个类的单输入模型（二进制分类）：
def single_layer_model():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 生成虚拟数据
    import numpy as np
    data = np.random.random((1000, 100))
    labels = np.random.randint(2, size=(1000, 1))

    # 训练模型，以 32 个样本为一个 batch 进行迭代
    model.fit(data, labels, epochs=10, batch_size=32)


def single_layer_model_ategorical():
    model = Sequential()
    # model.add(Dense(32, activation='relu', input_dim=100))
    model.add(Dense(32, activation='relu', input_dim=100))
    model.add(Dense(10, activation='softmax'))


    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 生成虚拟数据
    data = np.random.random((1000, 100))
    labels = np.random.randint(10, size=(1000, 1))

    # 将标签转换为分类的 one-hot 编码
    one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

    # 训练模型，以 32 个样本为一个 batch 进行迭代
    model.fit(data, one_hot_labels, epochs=10, batch_size=32)


# 基于多层感知器的softmax多分类

def seq_multi_model():
    x_train = np.random.random((1000, 20))
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
    x_test = np.random.random((100, 20))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=20))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=20,
              batch_size=128)
    score = model.evaluate(x_test, y_test, batch_size=128)
    print(score)

# binary_crossentropy MLP的二分类：
def seq_multi_model_mlp():
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
    model.fit(x_train, y_train,
              epochs=20,
              batch_size=128)
    score = model.evaluate(x_test, y_test, batch_size=128)
    print(score)

# 类似VGG的卷积神经网络：
def seq_multi_model_vgg():
    x_train = np.random.random((100, 100, 100, 3))
    print(x_train)
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
    x_test = np.random.random((20, 100, 100, 3))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)
    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
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

    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    model.fit(x_train, y_train, batch_size=32, epochs=10)
    score = model.evaluate(x_test, y_test, batch_size=32)

    print(score)


def seq_multi_model_lstm():
    data_dim = 16
    timesteps = 8
    num_classes = 10

    # 期望输入数据尺寸: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(32, return_sequences=True,
                   input_shape=(timesteps, data_dim)))  # 返回维度为 32 的向量序列
    model.add(LSTM(32, return_sequences=True))  # 返回维度为 32 的向量序列
    model.add(LSTM(32))  # 返回维度为 32 的单个向量
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # 生成虚拟训练数据
    x_train = np.random.random((1000, timesteps, data_dim))
    y_train = np.random.random((1000, num_classes))

    # 生成虚拟验证数据
    x_val = np.random.random((100, timesteps, data_dim))
    y_val = np.random.random((100, num_classes))

    model.fit(x_train, y_train,
              batch_size=64, epochs=5,
              validation_data=(x_val, y_val))
    score = model.evaluate(x_val, y_val, batch_size=16)
    print(score)


# 有问题没调试好
def seq_multi_model_lstm2():
    data_dim = 16
    timesteps = 8
    num_classes = 10
    max_features = 1024
    model = Sequential()
    model.add(Embedding(max_features, output_dim=data_dim))
    model.add(LSTM(data_dim))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='sigmoid'))
    inputs = Input(name='inputs', shape=[max_features])
    model = Model(inputs=inputs, outputs=model)  # 建立模型
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    # 生成虚拟训练数据
    x_train = np.random.random((1000, timesteps, data_dim))
    y_train = np.random.random((1000, num_classes))

    # 生成虚拟验证数据
    x_val = np.random.random((100, timesteps, data_dim))
    y_val = np.random.random((100, num_classes))
    model.fit(x_train, y_train, batch_size=16, epochs=10)
    score = model.evaluate(x_val, y_val, batch_size=16)

    print(score)

def  seq_multi_model_stateful():
    data_dim = 16
    timesteps = 8
    num_classes = 10
    batch_size = 32

    # 期望输入数据尺寸: (batch_size, timesteps, data_dim)
    # 请注意，我们必须提供完整的 batch_input_shape，因为网络是有状态的。
    # 第 k 批数据的第 i 个样本是第 k-1 批数据的第 i 个样本的后续。
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, stateful=True,
                   batch_input_shape=(batch_size, timesteps, data_dim)))
    model.add(LSTM(32, return_sequences=True, stateful=True))
    model.add(LSTM(32, stateful=True))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # 生成虚拟训练数据
    x_train = np.random.random((batch_size * 10, timesteps, data_dim))
    y_train = np.random.random((batch_size * 10, num_classes))

    # 生成虚拟验证数据
    x_val = np.random.random((batch_size * 3, timesteps, data_dim))
    y_val = np.random.random((batch_size * 3, num_classes))

    model.fit(x_train, y_train,
              batch_size=batch_size, epochs=5, shuffle=False,
              validation_data=(x_val, y_val))

    score = model.evaluate(x_val, y_val, batch_size=batch_size)

    print(score)

if __name__ == '__main__':
    seq_multi_model_lstm2()

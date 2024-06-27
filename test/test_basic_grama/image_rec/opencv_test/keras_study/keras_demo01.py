import keras
from keras import Sequential
from keras.src.layers import Dense
import numpy as np

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

if __name__ == '__main__':
    create_seq_model()
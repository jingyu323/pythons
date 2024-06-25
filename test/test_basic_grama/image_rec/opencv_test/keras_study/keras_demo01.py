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

if __name__ == '__main__':
    create_model()
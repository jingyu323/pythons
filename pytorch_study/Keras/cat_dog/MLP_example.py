import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential, Input
from keras.src.layers import LSTM, Conv1D, MaxPooling1D, Flatten
from keras.src.optimizers import Adam
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# 生成合成数据


# 也称为多层感知器，MLP
np.random.seed(0)
X = np.linspace(0, 10, 100)[:, np.newaxis]
y = np.sin(X) + 0.1 * np.random.randn(100, 1)
def demo_MLP():
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 创建模型
    model = Sequential()
    # 添加输入层和第一个隐藏层
    model.add(Dense(50, activation='relu', input_shape=(X_train.shape[1],)))
    # 添加第二个隐藏层
    model.add(Dense(20, activation='relu'))
    # 添加输出层
    model.add(Dense(1))
    # 打印模型概要
    model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test), verbose=2)
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)
    print(f'Test MAE: {test_mae}')
    y_pred = model.predict(X_test)
    plt.scatter(X_test, y_test, label='Actual')
    plt.plot(X_test, y_pred, 'r', label='Predicted')
    plt.legend()
    plt.show()

# LSTM（长短期记忆网络）是一种特殊的RNN（循环神经网络）架构，非常适合处理和预测时间序列数据。
def demo_LSTM():
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # 重构数据以适应LSTM输入
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    # 创建模型
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    # 训练模型
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=2)
    print(X_train.shape, X_test.shape)
    # 预测
    y_pred = model.predict(X_test)
    # 反标准化
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test)
    # 可视化结果
    plt.plot(X_test.reshape((20, 1)), y_test, label='Actual')
    plt.plot(X_test.reshape((20, 1)), y_pred, 'r', label='Predicted')
    plt.legend()
    plt.show()


def demo_cnn():
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # 重构数据以适应CNN输入
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    # 创建模型
    model = Sequential()

    print(X_train.shape[1], X_train.shape[2])
    print(X_train,X_train)

    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])),Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    # 训练模型
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=2)
    # 预测
    y_pred = model.predict(X_test)
    # 反标准化
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test)
    # 可视化结果
    plt.plot(X_test, y_test, label='Actual')
    plt.plot(X_test, y_pred, 'r', label='Predicted')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    demo_cnn()
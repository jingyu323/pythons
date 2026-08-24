# 1. 准备数据
import numpy as np
from keras import Sequential, Input
from keras.src import optimizers
from keras.src.datasets import mnist
from keras.src.layers import Dense, Flatten
from keras.src.optimizers import SGD
from keras.src.utils import to_categorical
from matplotlib import pyplot as plt


def demo1():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train, y_train)
    print(x_test, y_test)
    x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255
    x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # 2. 定义模型
    model = Sequential([
        Flatten(input_shape=(28 * 28,)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # 3. 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 4. 训练模型
    model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1)
    model.summary()

    # 5. 评估模型
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f'Test accuracy: {accuracy:.4f}')

    model.predict(x_test)

def demo2():
    X = np.random.randint(0, 100, (200, 3))
    a = [[5], [1.5], [1]]
    print(a)
    Y = np.mat(X) * np.mat(a) + np.random.normal(0, 0.05, (200, 1))  # 假定函数关系为y=w[x]+b
    X_train, y_train = X[:160], Y[:160]
    X_test, y_test = X[160:], Y[160:]
    model = Sequential()

    model.add(Dense(3 ,activation='sigmoid',input_dim=3))
    # model.add(Dense(10, activation='softmax'))
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizers.Adam(learning_rate=0.001, beta_2=0.999), loss="mse")
    # 训练模型
    history = model.fit(X_train, y_train, epochs=3000)
    cost = model.evaluate(X_test, y_test)
    print(f'Test cost: {cost}')
    print(model.layers[0].get_weights())
    plt.plot(range(3000), history.history["loss"])
    plt.show()

def demo3():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("xshape", x_train.shape)
    print(x_train)
    # 从本地路径加载MNIST数据集
    # local_mnist_path="../lib/mnist.npz"


    #
    # with np.load(local_mnist_path, allow_pickle=True) as data:
    #     x_train, y_train = data['x_train'], data['y_train']
    #     x_test, y_test = data['x_test'], data['y_test']
    # # 对数据进行归一化处理

    x_train = x_train.reshape(x_train.shape[0],-1)/255.0
    x_test = x_test.reshape(x_test.shape[0],-1)/255.0
    # x_train, x_test = x_train / 255.0, x_test / 255.0
    # to onehot
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    # 创建784 个神经元 ,输出10 个神经元
    model = Sequential()
    model.add(Dense(10,bias_initializer='one', activation='softmax', input_shape=(28 * 28,)))

#     定义优化器
    sgd = SGD(learning_rate=0.2,  nesterov=True)

    # 模型编译 使用交叉熵
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']);
    # 模型训练
    model.fit(x_train, y_train, epochs=30,batch_size=32, verbose=1)
#   模型评估
    loss, accuracy = model.evaluate(x_test, y_test)
    print("loss:", loss)
    print("accuracy:", accuracy)




def demo4():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("xshape", x_train.shape)
    print(x_train)
    # 从本地路径加载MNIST数据集
    # local_mnist_path="../lib/mnist.npz"


    #
    # with np.load(local_mnist_path, allow_pickle=True) as data:
    #     x_train, y_train = data['x_train'], data['y_train']
    #     x_test, y_test = data['x_test'], data['y_test']
    # # 对数据进行归一化处理

    x_train = x_train.reshape(x_train.shape[0],-1)/255.0
    x_test = x_test.reshape(x_test.shape[0],-1)/255.0
    # x_train, x_test = x_train / 255.0, x_test / 255.0
    # to onehot
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    # 创建784 个神经元 ,输出10 个神经元
    model = Sequential()
    model.add(Dense(200,bias_initializer='one', activation='tanh', input_shape=(28 * 28,)))
    model.add(Dense(100,bias_initializer='one', activation='tanh'))
    model.add(Dense(10,bias_initializer='one', activation='softmax'))

#     定义优化器
    sgd = SGD(learning_rate=0.2,  nesterov=True)

    # 模型编译 使用交叉熵
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']);
    # 模型训练
    model.fit(x_train, y_train, epochs=30,batch_size=32, verbose=1)
#   模型评估
    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss:", loss)
    print("test accuracy:", accuracy)

    loss, accuracy = model.evaluate(x_train, y_train)
    print("train loss:", loss)
    print("train accuracy:", accuracy)

def demo5():
    pass


if __name__ == '__main__':
    demo5()
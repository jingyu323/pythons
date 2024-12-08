from keras import Sequential
from keras.src.datasets import mnist
from keras.src.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.src.optimizers import SGD, Adam
from keras.src.utils import to_categorical


def demo1():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("xshape", x_train.shape)
    print(x_train)


    x_train = x_train.reshape( -1,28,28,1) / 255.0
    x_test = x_test.reshape(-1,28,28,1) / 255.0
    # x_train, x_test = x_train / 255.0, x_test / 255.0
    # to onehot
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    # 创建784 个神经元 ,输出10 个神经元
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    #     定义优化器

    adam = Adam(learning_rate=0.0001)

    # 模型编译 使用交叉熵
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy']);
    # 模型训练
    model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=1)
    #   模型评估
    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss:", loss)
    print("test accuracy:", accuracy)

    loss, accuracy = model.evaluate(x_train, y_train)
    print("train loss:", loss)
    print("train accuracy:", accuracy)



if __name__ == '__main__':
    demo1()
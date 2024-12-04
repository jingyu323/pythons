# 1. 准备数据
from keras import Sequential
from keras.src.datasets import mnist
from keras.src.layers import Dense, Flatten
from keras.src.utils import to_categorical


(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train,y_train)
print(x_test,y_test)
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
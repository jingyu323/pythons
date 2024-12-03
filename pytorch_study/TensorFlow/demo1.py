# 1. 准备数据
import numpy as np
import tensorflow as tf
from keras import Sequential

from tensorflow.keras.layers import Dense

x_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
y_train = np.array([[0], [0], [1], [1]])  # 标签为0或1

# 2. 定义模型
model = Sequential([
    Dense(1, activation='sigmoid', input_shape=(2,))
])

# 3. 编译模型
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# 4. 训练模型
model.fit(x_train, y_train, epochs=100, verbose=1)

# 5. 预测
predictions = model.predict(x_train)
print(predictions)
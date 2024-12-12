import keras
# coding: utf-8

# In[ ]:

import numpy as np
from keras import Sequential, Input, Model
from keras.src.datasets import mnist
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical

# 数据处理

# In[ ]:

# 载入数据
(x_train,y_train),(x_test,y_test) = mnist.load_data()
# (60000,28,28)->(60000,28,28,1)
x_train = x_train.reshape(-1,28,28,1)/255.0
x_test = x_test.reshape(-1,28,28,1)/255.0
# 换one hot格式
y_train =  to_categorical(y_train,num_classes=10)
y_test =  to_categorical(y_test,num_classes=10)


# 序贯（Sequential）模型

# In[ ]:

# 定义序贯模型
model = Sequential()

# 第一个卷积层
# input_shape 输入平面
# filters 卷积核/滤波器个数
# kernel_size 卷积窗口大小
# strides 步长
# padding padding方式 same/valid
# activation 激活函数
model.add(Conv2D(
    input_shape = (28,28,1),
    filters = 32,
    kernel_size = 5,
    strides = 1,
    padding = 'same',
    activation = 'relu'
))
# 第一个池化层
model.add(MaxPooling2D(
    pool_size = 2,
    strides = 2,
    padding = 'same',
))
# 第二个卷积层
model.add(Conv2D(64,5,strides=1,padding='same',activation = 'relu'))
# 第二个池化层
model.add(MaxPooling2D(2,2,'same'))
# 把第二个池化层的输出扁平化为1维
model.add(Flatten())
# 第一个全连接层
model.add(Dense(1024,activation = 'relu'))
# Dropout
model.add(Dropout(0.5))
# 第二个全连接层
model.add(Dense(10,activation='softmax'))


# 函数式（Functional）模型

# In[ ]:

inputs = Input(shape=(28,28,1))
x = Conv2D(filters=32, kernel_size=5, padding='same', activation='relu')(inputs)
x = MaxPooling2D(pool_size = 2)(x)
x = Conv2D(filters=64, kernel_size=5, padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size = 2)(x)
x = Flatten()(x)
x = Dense(1024,activation = 'relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(10,activation='softmax')(x)
model = Model(inputs=inputs, outputs=predictions)


# 训练模型

# In[ ]:

# 定义优化器
adam = Adam(lr=1e-4)

# 定义优化器，loss function，训练过程中计算准确率
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

# 训练模型
model.fit(x_train,y_train,batch_size=64,epochs=10)

# 评估模型
loss,accuracy = model.evaluate(x_test,y_test)

print('test loss',loss)
print('test accuracy',accuracy)


# Inception:
# <center><img src="inception.jpg" alt="FAO" width="500"></center> 

# In[ ]:

input_img = Input(shape=(256, 256, 3))

tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

output =  concatenate([tower_1, tower_2, tower_3], axis=1)


# <h3 align = "center">欢迎大家关注我的公众号，或者加我的微信与我交流。</h3>
# <center><img src="wx.png" alt="FAO" width="300"></center> 

# In[ ]:




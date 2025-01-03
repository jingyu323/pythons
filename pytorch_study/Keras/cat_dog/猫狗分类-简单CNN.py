import os

import numpy as np
from keras import Sequential, Model
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.src.optimizers import Adam
from keras.src.saving import load_model

from keras_preprocessing.image import ImageDataGenerator


# coding: utf-8


# 定义模型
model = Sequential()
model.add(Conv2D(input_shape=(150,150,3), filters=32, kernel_size=3, strides=1, padding='same', activation = 'relu'))
model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))

model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation = 'relu'))
model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))

model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation = 'relu'))
model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))

model.add(Flatten())
model.add(Dense(64,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation = 'softmax'))

# 定义优化器
adam = Adam(learning_rate=1e-4)

# 定义优化器，loss function，训练过程中计算准确率
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()



# 训练集数据生成
datagen = ImageDataGenerator(
    rotation_range=40,
    rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)




# 测试集数据处理
test_datagen = ImageDataGenerator(rescale=1./255)


# flow_from_directory:  
# * directory: 目标文件夹路径,对于每一个类,该文件夹都要包含一个子文件夹.子文件夹中任何JPG、PNG、BNP、PPM的图片都会被生成器使用.详情请查看此脚本  
# * target_size: 整数tuple,默认为(256, 256). 图像将被resize成该尺寸  
# * color_mode: 颜色模式,为"grayscale","rgb"之一,默认为"rgb".代表这些图片是否会被转换为单通道或三通道的图片.  
# * classes: 可选参数,为子文件夹的列表,如['dogs','cats']默认为None. 若未提供,则该类别列表将从directory下的子文件夹名称/结构自动推断。每一个子文件夹都会被认为是一个新的类。(类别的顺序将按照字母表顺序映射到标签值)。通过属性class_indices可获得文件夹名与类的序号的对应字典。  
# * class_mode: "categorical", "binary", "sparse"或None之一. 默认为"categorical. 该参数决定了返回的标签数组的形式, "categorical"会返回2D的one-hot编码标签,"binary"返回1D的二值标签."sparse"返回1D的整数标签,如果为None则不返回任何标签, 生成器将仅仅生成batch数据, 这种情况在使用model.predict_generator()和model.evaluate_generator()等函数时会用到.  
# * batch_size: batch数据的大小,默认32  
# * shuffle: 是否打乱数据,默认为True  
# * seed: 可选参数,打乱数据和进行变换时的随机数种子  
# * save_to_dir: None或字符串，该参数能让你将提升后的图片保存起来，用以可视化  
# * save_prefix：字符串，保存提升后图片时使用的前缀, 仅当设置了save_to_dir时生效  
# * save_format："png"或"jpeg"之一，指定保存图片的数据格式,默认"jpeg"  
# * flollow_links: 是否访问子文件夹中的软链接  


batch_size = 32
# 生成训练数据
train_generator = datagen.flow_from_directory(
        'E:/data/kreas/Kaggle/cat-dog-small/train',  # 训练数据路径
        target_size=(150, 150),  # 设置图片大小
        class_mode='categorical',
        batch_size=batch_size # 批次大小
        )
print(train_generator)
# 测试数据
test_generator = test_datagen.flow_from_directory(
        'E:/data/kreas/Kaggle/cat-dog-small/test',  # 训练数据路径
        target_size=(150, 150),  # 设置图片大小
        class_mode='categorical',
        batch_size=batch_size # 批次大小
        )


# 统计文件个数
totalFileCount = sum([len(files) for root, dirs, files in os.walk('E:/data/kreas/Kaggle/cat-dog-small/train')])
print(totalFileCount)
# for X_batch, y_batch in train_generator:
#
#     print(X_batch.shape,y_batch.shape)



if os.path.exists("CNN1.keras"):
    model = load_model('CNN1.keras')
    print("CNN1.keras  exists")

else:
    model.fit(
            train_generator,
            steps_per_epoch=int(totalFileCount/batch_size),
            epochs=5,
            validation_data=test_generator,
            validation_steps=int(1000/batch_size),
            )
    # 保存模型
    model.save('CNN1.keras')


bottleneck_features_test = model.predict(test_generator, 30)


print(bottleneck_features_test)
test_loss, test_acc = model.evaluate(test_generator )

print(test_loss,test_acc)

# <h3 align = "center">欢迎大家关注我的公众号，或者加我的微信与我交流。</h3>
# <center><img src="wx.png" alt="FAO" width="300"></center> 






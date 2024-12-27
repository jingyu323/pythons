import cv2
#正则匹配使用：
import re
import os
import tensorflow as tf
#此库用于拷贝，删除，移动，复制以及解压缩
import shutil
import numpy as np
from keras.src.losses import SparseCategoricalCrossentropy
from keras_core.src.optimizers import Adam
from tf_keras import Input, Model
from tf_keras.src.layers import Dense, GlobalAvgPool2D, Dropout
from tf_keras.src.preprocessing.image import ImageDataGenerator

base_dir = 'E:/data/kreas/Kaggle/cat-dog-small/'
train_dir = os.path.join(base_dir, 'train/cats')
dataAll= ImageDataGenerator(validation_split=0.3)
genIteratorForTrain=dataAll.flow_from_directory(base_dir,batch_size=32,subset="training",class_mode='sparse')
#(32, 256, 256, 3)
print(genIteratorForTrain.next()[0].shape)
#(32,)
print(genIteratorForTrain.next()[1].shape)
#提取出所需的图片和label
#嵌套一层生成器:
#使用tf.data.Dataset不能使用无限生成器：
def Gen():
    for index,data in enumerate(genIteratorForTrain):
        #每批32的情况下，输出的批数，数量
        if index>genIteratorForTrain.__len__():
            return
        else:
            print(index)
            yield data
#输出图片的总数量
Gen()
print(genIteratorForTrain.n)

ds = tf.data.Dataset.from_generator(
    Gen,
    output_types=(tf.float32, tf.float32),
    #批次大小用None替代,自动推算,或是不出现,自动适配
    #output_shapes = ([None, 256, 256, 3],[None])
)

def mapfun(obj1,obj2):
    return obj1,obj2
#最大可用线程设置,拉满算力:
ds.map(mapfun,num_parallel_calls=tf.data.experimental.AUTOTUNE)

#(32, 256, 256, 3)批次自动按ImageDataGenerator进行设置
print(next(iter(ds))[0].shape)
#(32,)
print(next(iter(ds))[1].shape)
#17花朵分类的输入形状为(256, 256, 3)
shape_in=genIteratorForTrain.next()[0].shape[1:]
model=Dense(100)
#定义一个输入层：
inputs=Input(shape=shape_in)
#此处x形状为(None, 7, 7, 2048)
x=model(inputs)
#此处x形状为(None, 2048)
x=GlobalAvgPool2D()(x)
#此处x形状为(None, 2048)
x=Dense(units=17)(x)
#此处x形状为(None, 2048)
x=Dropout(rate=0.5)(x)
model2=Model(inputs = inputs, outputs = x)
model2.compile(optimizer=Adam(learning_rate=0.001),loss=SparseCategoricalCrossentropy(from_logits=True),metrics=['sparse_categorical_accuracy'])
history=model2.fit(x=ds,epochs=10,batch_size=32)

import numpy
import os

from keras import applications, Model
from keras.api import optimizers
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.layers import Flatten, Dense
from keras.src.legacy.preprocessing.image import ImageDataGenerator

files_train = 0
files_validation = 0

cwd = os.getcwd()
folder = 'train_data/train'

for sub_folder in os.listdir(folder):
    path, dirs, files = next(os.walk(os.path.join(folder,sub_folder)))
    files_train += len(files)


folder = 'train_data/test'
for sub_folder in os.listdir(folder):
    path, dirs, files = next(os.walk(os.path.join(folder,sub_folder)))
    files_validation += len(files)

print(files_train,files_validation)

img_width, img_height = 48, 48
train_data_dir = "train_data/train"
validation_data_dir = "train_data/test"
nb_train_samples = files_train
nb_validation_samples = files_validation
batch_size = 32
epochs = 15
num_classes = 2

model = applications.VGG16(weights='imagenet', include_top=False, input_shape = (img_width, img_height, 3))


for layer in model.layers[:10]:
    layer.trainable = False


x = model.output
x = Flatten()(x)
predictions = Dense(num_classes, activation="softmax")(x)


model_final = Model(inputs = model.input, outputs = predictions)


model_final.compile(loss = "categorical_crossentropy",
                    optimizer = optimizers.SGD(lr=0.0001, momentum=0.9),
                    metrics=["accuracy"])


train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.1,
width_shift_range = 0.1,
height_shift_range=0.1,
rotation_range=5)

test_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.1,
width_shift_range = 0.1,
height_shift_range=0.1,
rotation_range=5)

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size,
class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
validation_data_dir,
target_size = (img_height, img_width),
class_mode = "categorical")

checkpoint = ModelCheckpoint("car1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')




history_object = model_final.fit_generator(
train_generator,
epochs = epochs,
steps_per_epoch=32,

validation_data = validation_generator,
validation_steps = 52,
callbacks = [checkpoint, early])

# history_object = model_final.fit_generator(
#     train_generator,
#     samples_per_epoch=nb_train_samples,
#     epochs=epochs,
#     validation_data=validation_generator,
#     nb_val_samples=nb_validation_samples,
#     callbacks=[checkpoint, early])
"""
generator：生成器函数，生成器的输出应该为：
steps_per_epoch：整数，当生成器返回steps_per_epoch次数据时计一个epoch结束，执行下一个epoch

epochs：整数，数据迭代的轮数
生成验证集的生成器

一个形如（inputs,targets）的tuple

一个形如（inputs,targets，sample_weights）的tuple

validation_steps: 当validation_data为生成器时，本参数指定验证集的生成器返回次数

eps_per_epoch来实现的，每次生产的数据就是一个batch，因此steps_per_epoch的值我们通过会设为（样本数/batch_size）
。如果我们的generator是sequence类型，那么这个参数是可选的，默认使用len(generator) 。

"""
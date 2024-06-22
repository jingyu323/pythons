

from PIL import Image
import numpy as np
from keras import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import SGD


def process_x(path):
    img = Image.open(path)
    img = img.resize((96, 96))
    img = img.convert('RGB')
    img = np.array(img)

    img = np.asarray(img, np.float32) / 255.0
    # 也可以进行进行一些数据数据增强的处理
    return img


def generate_arrays_from_file(x_y):
    # x_y 是我们的训练集包括标签，每一行的第一个是我们的图片路径，后面的是图片标签

    global count
    batch_size = 8
    while 1:
        batch_x = x_y[(count - 1) * batch_size:count * batch_size, 0]
        batch_y = x_y[(count - 1) * batch_size:count * batch_size, 1:]

        batch_x = np.array([process_x(img_path) for img_path in batch_x])
        batch_y = np.array(batch_y).astype(np.float32)
        print("count:" + str(count))
        count = count + 1
        yield batch_x, batch_y


model = Sequential()
model.add(Dense(units=1000, activation='relu', input_dim=2))
model.add(Dense(units=2, activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
count = 1
x_y = ["../image/train/*.jpg", "../image/test/*.jpg"]
model.fit_generator(generate_arrays_from_file(x_y), steps_per_epoch=10, epochs=2, max_queue_size=1, workers=1)

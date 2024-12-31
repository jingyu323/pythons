from keras_core import Sequential
from keras_core.src.layers import Dense, Dropout
from keras_core.src.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report#综合结果对比

from keras import initializers#初始化权重参数
from keras import regularizers#正则化
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import cv2
import os

from Keras.cat_dog import utils_paths

#--dataset --model --label-bin --plot
# 输入参数
print("[INFO] 开始读取数据")
data = []
labels = []

# 拿到图像数据路径，方便后续读取
imagePaths = sorted(list(utils_paths.list_images("E:/data/kreas/Kaggle/cat-dog-small/")))
random.seed(42)
random.shuffle(imagePaths)

# 遍历读取数据
for imagePath in imagePaths:
	# 读取图像数据，由于使用神经网络，需要给定成一维
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (32, 32)).flatten()
	data.append(image)

	# 读取标签
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

# scale图像数据
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# 数据集切分
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# 转换标签，one-hot格式
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# 网络模型结构：3072-512-256-3
model = Sequential()
# kernel\_regularizer=regularizers.l2(0.01)
# keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
# initializers.random\_normal
# #model.add(Dropout(0.8))
model.add(Dense(512, input_shape=(3072,), activation="relu" ,kernel_initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(256, activation="relu",kernel_initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(len(lb.classes_), activation="softmax",kernel_initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),kernel_regularizer=regularizers.l2(0.01)))

# 初始化参数
INIT_LR = 0.001
EPOCHS = 200

# 给定损失函数和评估方法
print("[INFO] 准备训练网络...")
opt = SGD(learning_rate=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# 训练网络模型
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=EPOCHS, batch_size=32)

# 测试网络模型
print("[INFO] 正在评估模型")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# 当训练完成时，绘制结果曲线
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
#plt.plot(N[150:], H.history["loss"][150:], label="train\_loss")
#plt.plot(N[150:], H.history["val\_loss"][150:], label="val\_loss")
plt.plot(N[150:], H.history["accuracy"][150:], label="train_acc")
plt.plot(N[150:], H.history["val_accuracy"][150:], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("cat_train ")

# 保存模型到本地
print("[INFO] 正在保存模型")
model.save("cat_train.h5")
f = open( "label_bin", "wb")
f.write(pickle.dumps(lb))
f.close()


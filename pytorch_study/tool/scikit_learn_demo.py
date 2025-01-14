import os

import cv2
import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from Keras.cat_dog import utils_paths
def  irs():
    # 加载Iris数据集
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print(X_train)
    print(X_test)

def transform():
    # 创建转换器实例
    encoder = OneHotEncoder(handle_unknown='ignore')

    # 训练数据
    train_data = [['low', 'age'], ['medium', 'age'], ['high', 'age']]

    # 测试数据
    test_data = [['medium', 'income'], ['high', 'income']]

    # 使用.fit_transform()拟合训练数据并转换
    encoded_train = encoder.fit_transform(train_data)

    # 使用.transform()只转换测试数据
    encoded_test = encoder.transform(test_data)

    # 打印结果
    print("Encoded Train Data:\n", encoded_train)
    print("Encoded Test Data:\n", encoded_test)

    train_data2 = ['low', 'medium', 'high']
    test_data = ['medium',  'high', 'income']

    lb = LabelBinarizer()
    trainY = lb.fit_transform(train_data2)
    testY = lb.transform(test_data)

    print("Encoded trainY Data:\n", trainY)
    print("Encoded testY Data:\n", testY)


def transform2():
    # --dataset --model --label-bin --plot
    # 输入参数
    print("[INFO] 开始读取数据")
    data = []
    labels = []

    # 拿到图像数据路径，方便后续读取
    imagePaths = sorted(list(utils_paths.list_images("E:/data/kreas/Kaggle/cat-dog-small/train")))
    random.seed(42)
    random.shuffle(imagePaths)
    # 遍历读取数据
    for imagePath in imagePaths:
        # 读取图像数据，由于使用神经网络，需要输入数据给定成一维
        image = cv2.imread(imagePath)
        # 而最初获取的图像数据是三维的，则需要将三维数据进行拉长
        image = cv2.resize(image, (32, 32)).flatten()
        data.append(image)

        # 读取标签，通过读取数据存储位置文件夹来判断图片标签
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

    # scale图像数据，归一化
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # 数据集切分
    (trainX, testX, trainY, testY) = train_test_split(data,
                                                      labels, test_size=0.25, random_state=42)

    # 转换标签，one-hot格式
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)

    print(trainX,trainY,testX,testY)

def convert_to_one_hot(labels, num_classes):
    #计算向量有多少行
    num_labels = len(labels)
    #生成值全为0的独热编码的矩阵
    labels_one_hot = np.zeros((num_labels, num_classes))
    #计算向量中每个类别值在最终生成的矩阵“压扁”后的向量里的位置
    index_offset = np.arange(num_labels) * num_classes
    #遍历矩阵，为每个类别的位置填充1
    labels_one_hot.flat[index_offset + labels] = 1
    return labels_one_hot
def catlog():
    b = [2, 4, 6, 8, 6, 2, 3, 7]
    print(convert_to_one_hot(b, 9))


if __name__ == '__main__':
    catlog()


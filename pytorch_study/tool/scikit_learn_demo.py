import os

import cv2
import numpy
import torch
from pandas import Series
from sklearn import datasets
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from Keras.cat_dog import utils_paths
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import  OneHotEncoder
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

    # testdata = pd.DataFrame({'pet': ['cat', 'dog', 'dog', 'fish'], 'age': [4, 6, 3, 3],
    #                          'salary': [4, 5, 1, 1]})
    #
    # print(testdata.age)
    # print(testdata.salary)

    # OneHotEncoder( handle_unknown='ignore').fit_transform( testdata.age)  # testdata.age 这里与 testdata[['age']]等价

    # a1 = OneHotEncoder( ).fit_transform(pd.Series(testdata[['age']]))
    # a2 = OneHotEncoder( ).fit_transform(pd.Series(testdata[['salary']]))
    # final_output = numpy.hstack((a1, a2))
    # print(final_output)

    data = {'degree': ['master', 'master', 'PHD'], 'grade': ['A', 'B', 'C']}
    df = pd.DataFrame(data)

    enc = OneHotEncoder()
    enc.fit(df)
    print(enc.categories_)
    print( enc.get_feature_names_out())
    print( enc.feature_names_in_)
    print( enc._check_feature_names( ))


def onehot_demo():
    encoder = OneHotEncoder()
    encoder.fit([
        [0, 2, 1, 12],
        [2, 3, 5, 3],
        [1, 3, 2, 12],
        [1, 2, 4, 3]
    ])
    encoded_vector = encoder.transform([[2, 3, 5, 3]]).toarray()
    print("\n Encoded vector =", encoded_vector)

    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit([['体育', '军事'],
                 ['计科', '开心'],
                 ['娱乐', '军事']])
    encoded_vector = encoder.transform([['计科', '开心']]).toarray()
    print(encoder.inverse_transform([[0, 0, 1, 0, 1]]))

    print("\n Encoded vector =", encoded_vector)


    # df = pd.DataFrame({
    #     "A": ['男', '男', '女', '女', '其它'],
    #     "B": [100, 200, 300, 400, 500]
    # })
    # # OneHotEncoder中的sparse参数被设置为了False，它可以控制转换后的稠密性，即是否产生稀疏矩阵。
    # encoder = OneHotEncoder(handle_unknown='ignore')
    # gender_encoded = encoder.transform(df['A'].values.reshape(-1, 1)).toarray()
    # print(gender_encoded)
    # df.assign(oneencode=gender_encoded )

def one_hot():
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit([['体育', '军事'],
                 ['计科', '开心'],
                 ['娱乐', '军事']])
    encoded_vector = encoder.transform([['计科', '难过']]).toarray()
    print("\n Encoded vector =", encoded_vector)
    print(encoder.categories_)


if __name__ == '__main__':
    # onehot_demo()
    # one_hot()
    catlog()


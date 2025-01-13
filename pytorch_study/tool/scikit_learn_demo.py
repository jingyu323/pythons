import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder

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

if __name__ == '__main__':
    transform()


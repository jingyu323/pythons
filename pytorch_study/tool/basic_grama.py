import glob
from sklearn.linear_model import LinearRegression

import numpy as np
from keras_preprocessing.image import ImageDataGenerator

# np.ravel()
# 和np.flatten()
# 都会返回一维数组，但它们在处理内存时有所不同。np.ravel()
# 返回的是原数组的视图（view），而np.flatten()
# 返回的是原数组的副本（copy）。
# np.ndarray.resize()
# 会直接改变原数组的形状和大小，而不是返回一个新数组。
# 读取目录
directory = '../Keras/'
for filename in glob.glob(directory + '/*'):
    print(filename)

# 假设我们有一维的特征数组和目标变量数组
X_1d = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 6, 8, 10])

# 创建线性回归模型
model = LinearRegression()

# 拟合模型，注意X_1d已经是二维数组，满足模型输入要求
model.fit(X_1d, y)

# 使用模型进行预测
X_new = np.array([6, 7]).reshape(-1, 1)
print(X_new)
y_pred = model.predict(X_new)

print("预测结果:")
print(y_pred)
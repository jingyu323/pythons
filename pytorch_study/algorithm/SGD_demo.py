# 使用步骤
# 导入数据和库：
import numpy as np
import matplotlib.pyplot as plt

# 生成一些随机数据
np.random.seed(42)
x_train = 2 * np.random.rand(100, 1)
y_train = 4 + 3 * x_train + np.random.randn(100, 1)

print(x_train,y_train)

# 初始化模型参数： 为模型参数赋初始值。假设我们要训练一个简单的线性回归模型 ( y = w x + b ) ，初始参数可以设为0或随机值
w = np.random.randn()
b = np.random.randn()


# 设置学习率和超参数： 设定学习率和其他超参数。例如：
learning_rate = 0.01
num_epochs = 1000

# 定义损失函数： 定义我们要最小化的损失函数，比如均方误差（MSE）。
def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
# 定义梯度计算： 根据损失函数定义梯度的计算方法。
def compute_gradients(x, y, w, b):
    y_pred = w * x + b
    dw = -2 * np.mean(x * (y - y_pred))
    db = -2 * np.mean(y - y_pred)
    return dw, db

# SGD更新步骤： 根据随机选择的样本计算梯度并更新模型参数。以下是循环内的实现方式：
for epoch in range(num_epochs):
    # 随机选择一个样本
    idx = np.random.randint(len(x_train))
    x_sample = x_train[idx]
    y_sample = y_train[idx]

    # 计算梯度
    dw, db = compute_gradients(x_sample, y_sample, w, b)

    # 更新参数
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # 打印损失信息
    if epoch % 100 == 0:
        y_pred = w * x_train + b
        loss = compute_loss(y_train, y_pred)
        print(f'Epoch {epoch}, Loss: {loss}')

# 模型验证和评估： 在训练完成后，可以使用验证集或测试集来评估模型的性能。例如：

# 模型验证和评估
x_test = np.array([[1], [2]])
y_test = 4 + 3 * x_test
y_test_pred = w * x_test + b


y_test_pred = w * x_test + b
test_loss = compute_loss(y_test, y_test_pred)
print(f'Test Loss: {test_loss}')

# 绘制拟合结果
plt.scatter(x_train, y_train, color='blue', label='Training data')
plt.plot(x_test, y_test_pred, color='red', label='Fitted line')
plt.legend()
plt.show()
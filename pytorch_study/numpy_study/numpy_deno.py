import numpy as np

np.random.seed(42)
x_train = 2 * np.random.rand(10, 2)
y_train = 4 + 3 * x_train + np.random.randn(10, 2)

print(x_train)
print(y_train)


arr = np.array([1, 2, 3, 4, 5])

print(arr)
print(np.sum(arr))      # 求和：15

# 数组比较

mask = arr > 3

print(mask)          # [False False False True True]

print(arr[mask])     # [4 5]  只保留大于3的元素

# 创建随机数组

arr = np.random.randint(1, 101, size=(5, 5))

print("随机数组：\n", arr)

# 找最大值位置

max_index = np.unravel_index(arr.argmax(), arr.shape)

print("最大值位置：", max_index)

# 计算行平均值

row_means = arr.mean(axis=1)

print("每行平均值：", row_means)
import numpy as np

np.random.seed(42)
x_train = 2 * np.random.rand(10, 2)
y_train = 4 + 3 * x_train + np.random.randn(10, 2)

print(x_train)
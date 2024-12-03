
Keras
Keras本身并不是一个框架，而是一个位于其他深度学习框架之上的高级API。目前它支持TensorFlow、Theano和CNTK。

# 定义模型的类与函数

# 张量和计算图模型与标准数组的比较 

# 训练模型
```
history = model.fit_generator(
    generator=train_generator,
    epochs=10,
    validation_data=validation_generator)
```
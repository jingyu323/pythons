
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
# (4)控制CPU与GPU模式的比较

```python
with tf.device('/cpu:0'):
    y = apply_non_max_suppression(x)

# 模型训练步骤

```
## 卷积
padding：在原始图像的边缘用了像素填充
```python
#keras
output = Conv2D(input.shape[-1] // reduction, kernel = (1,1), padding = "valid", use_bias = False)(output)
```

## 反卷积
```python

#keras
output = Conv2DTranspose(out_channels, kernel_size, stride, use_bias=bias)(input)
```

## 上采样 
自定义

## 池化 

## BatchNormalization 

## 激活

## 训练 

Keras
Keras本身并不是一个框架，而是一个位于其他深度学习框架之上的高级API。目前它支持TensorFlow、Theano和CNTK。

Keras 帮我们把这些“指令”组织起来，让电脑更容易学习。 它就像一个好老师，让电脑学习得更快、更有效率。 所以，Keras 帮助电脑学习各种事情，比如识别图片、预测天气等等。

### 核心特点

简洁易用：提供了非常直观的API，用户可以快速上手，适合新手和中小型项目。
高度模块化：允许用户自由组合层、优化器、损失函数等，模型的可读性和可维护性较高。
与TensorFlow完美结合：在TensorFlow 2.x之后，Keras成为TensorFlow的官方高级API，集成更为紧密。

<table><thead><tr><th>知识点</th><th>描述</th></tr></thead><tbody><tr><td>Sequential模型</td><td>一种按顺序堆叠网络层的模型。</td></tr><tr><td>函数式模型</td><td>用于构建更复杂的模型，支持分支和合并等操作。</td></tr><tr><td>编译模型</td><td>使用<code>.compile</code>方法指定损失函数、优化器和评估指标。</td></tr><tr><td>训练模型</td><td>使用<code>.fit()</code>方法在训练数据上进行迭代训练。</td></tr></tbody></table>

### 应用场景：

快速原型开发和中小型项目，特别是在自然语言处理和图像处理任务中。

### 核心组件：

1. Sequential：顺序模型，用于搭建简单的神经网络。
2. Model：函数式模型，用于搭建复杂的神经网络。
3. layers：网络层模块，提供卷积层、全连接层等。

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

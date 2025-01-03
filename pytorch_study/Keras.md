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

### 常用方式

- max-pooling 
- mean-pooling 
- stochastic pooling 



## BatchNormalization 



## 激活激活函数

#### Softmax

Softmax是一种常用的激活函数，主要用于多分类问题中，可以将输入的神经元转化为概率分布。它的主要特点是输出值范围在0-1之间，且所有输出值的总和为1。

使用场景：

- 在多分类任务中，用于将神经网络的输出转换为概率分布。
- 在自然语言处理、图像分类、语音识别等领域广泛应用。

优点：在多分类问题中，能够为每个类别提供一个相对的概率值，方便后续的决策和分类。

缺点：会出现梯度消失或梯度爆炸问题。

优化方案：

- **使用ReLU等其他激活函数：**结合使用其他激活函数，如ReLU或其变种（Leaky ReLU和Parametric ReLU）。
- **使用深度学习框架中的优化技巧：**利用深度学习框架（如TensorFlow或PyTorch）提供的优化技巧，如批量归一化、权重衰减等。

#### ReLU函数

简介：ReLU激活函数是一种简单的非线性函数，其数学表达式为f(x) = max(0, x)。当输入值大于0时，ReLU函数输出该值；当输入值小于或等于0时，ReLU函数输出0



![](E:\study\git\pythons\pytorch_study\Keras\ReLU.jpg)



函数图像

![](.\Keras\ReLU_res.png)

使用场景：ReLU激活函数广泛应用于深度学习模型中，尤其在卷积神经网络（CNN）中。它的主要优点是计算简单、能有效缓解梯度消失问题，并能够加速模型的训练。因此，在训练深度神经网络时，ReLU常常作为首选的激活函数。

优点：

- **缓解梯度消失问题：**与Sigmoid和Tanh等激活函数相比，ReLU在激活值为正时不会使梯度变小，从而避免了梯度消失问题。
- **加速训练：**由于ReLU的简单性和计算高效性，它可以显著加速模型的训练过程。

缺点：

- **“死亡神经元”问题：**当输入值小于或等于0时，ReLU的输出为0，导致该神经元失效，这种现象称为“死亡神经元”。
- **不对称性：**ReLU的输出范围是[0, +∞)，而输入值为负数时输出为0，这导致ReLU输出的分布不对称，限制了生成的多样性。

优化方案：

- **Leaky ReLU：**Leaky ReLU在输入小于或等于0时，输出一个较小的斜率，避免了完全的“死亡神经元”问题。
- **Parametric ReLU（PReLU）：**与Leaky ReLU不同的是，PReLU的斜率不是固定的，而是可以根据数据进行学习优化。



#### Sigmoid函数

函数

简介：Sigmoid函数是一种常用的非线性函数，可以将任何实数映射到0到1之间。它通常用于将不归一化的预测值转换为概率分布。

![](E:\study\git\pythons\pytorch_study\Keras\Sigmoid.jpg)

函数图像

![](.\Keras\Sigmoid_res.png)

使用场景：

- 输出限制在0到1之间，表示概率分布。
- 处理回归问题或二元分类问题。

优点：

- 可以将任何范围的输入映射到0-1之间，适合表示概率。
- 这个范围是有限的，这使得计算更加简单和快速。

缺点：在输入值非常大时，梯度可能会变得非常小，导致梯度消失问题。

优化方案：

- **使用ReLU等其他激活函数：**结合使用其他激活函数，如ReLU或其变种（Leaky ReLU和Parametric ReLU）。
- **使用深度学习框架中的优化技巧：**利用深度学习框架（如TensorFlow或PyTorch）提供的优化技巧，如梯度裁剪、学习率调整等。

#### Tanh函数

函数

简介：Tanh函数是Sigmoid函数的双曲版本，它将任何实数映射到-1到1之间。

![](E:\study\git\pythons\pytorch_study\Keras\Tanh.jpg)

函数图像

![](.\Keras\Tanh_res.png)

使用场景：当需要一个比Sigmoid更陡峭的函数，或者在某些需要-1到1范围输出的特定应用中。

优点：提供了更大的动态范围和更陡峭的曲线，可以加快收敛速度。

缺点：Tanh函数的导数在输入接近±1时迅速接近于0，导致梯度消失问题。

优化方案：

- **使用ReLU等其他激活函数：**结合使用其他激活函数，如ReLU或其变种（Leaky ReLU和Parametric ReLU）。
- **采用残差连接：**残差连接是一种有效的优化策略，如ResNet（残差网络）。





| 激活函数 | 优点                                                         | 缺点                                                         |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Softmax  | 在多分类问题中，能够为每个类别提供一个相对的概率值，方便后续的决策和分类。 | 会出现梯度消失或梯度爆炸问题。                               |
| ReLU     | **缓解梯度消失问题：**与Sigmoid和Tanh等激活函数相比，ReLU在激活值为正时不会使梯度变小，从而避免了梯度消失问题。<br /> **加速训练：**由于ReLU的简单性和计算高效性，它可以显著加速模型的训练过程。 | **“死亡神经元”问题：**当输入值小于或等于0时，ReLU的输出为0，导致该神经元失效，这种现象称为“死亡神经元”。 <br />**不对称性：**ReLU的输出范围是[0, +∞)，而输入值为负数时输出为0，这导致ReLU输出的分布不对称，限制了生成的多样性。 |
|          |                                                              |                                                              |

### 

## 训练 

### 拟合合适

### 过拟合

- Early stopping
 一般的做法是记录目前为止最好的validdation accuracy 当连续10个epoch 没达到最佳accurancy时则可以认为accuracy不再提高了，因此便可以停止迭代
- 正则化
  - L1 正则
  - L2 正则
- 降低学习率
- 

### 欠拟合：增加网络容量来达到过拟合，多增加模型的层数


## 优化函数

### 梯度下降法

#### 标准梯度下降法
计算所有样本汇总误差，然后根据总误差来更新权值 

#### 随机梯度下降法
随机抽取样本计算误差，然后更新权值

#### 批量梯度下降法

### adagrad 

优势在于不需要 认为的调节学习率，他可以自动调节。
缺点在于随着迭代次数的增加学习率也会越来越低，最终趋向于0 


### RMSprop
RMSprop 是 adagrad 改进，不会出现学习率越来越低的问题，自己会调节学习率

### Adadelta 

### Adam

是一种常见的优化器。Adam会存储之前的衰减的平方梯度，同时它会保存之前的衰减梯度。经过一些处理之后再来更新权值W。

### NAG

## CNN
并广泛于图像处理和NLP等领域的一种多层神经网络

- CNN 通过局部视野感受和权值共享减少了神经网络需要训练的参数个数

### 卷积核

###  池化 

####  常用方式

- max-pooling 
- mean-pooling 
- stochastic pooling 

#### 卷积padding

- samp padding
给平面外部补0，得到跟原来大小相同的平面
- valid padding
得到一个比原来平面小的平面

## 神经网络

### LeNET-5 
是最早的卷积神经网络之一，对手写数字识别正确率在99%以上

##  RNN  循环神经网络或者递归神经网络
在自然语言，语言识别，翻译以及图像处理等=领域有着非常好的应用

### LSTM  长时间记忆网络

是一种特殊的RNN 网络 ，适用于决策的神经网络应用



对抗神经网络：

**cyclegan**： 实现图片中对象的替换



# 需要解决的问题：

1. fit 和 fit_generator  互相转换的问题
版本合适根本不需要转换 image genrator 直接能使用


pip install tensorflow-directml-plugin


pip install -U tensorflow-gpu==2.10.0
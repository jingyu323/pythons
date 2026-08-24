
Pytorch
Pytorch是一个深度学习框架(类似于TensorFlow)，由Facebook的人工智能研究小组开发。与Keras一样，它也抽象出了深层网络编程的许多混乱部分。

就高级和低级代码风格而言，Pytorch介于Keras和TensorFlow之间。比起Keras具有更大的灵活性和控制能力，但同时又不必进行任何复杂的声明式编程(declarative programming)。


## 特性
核心特点
动态计算图：PyTorch支持动态图机制，允许在运行时动态修改模型结构，非常适合实验和研究。
强大的社区支持：PyTorch拥有丰富的文档和社区资源，适合开发者快速入门和进行复杂项目开发。
GPU加速：支持GPU加速，提升模型训练速度。

<div class="table-box"><table><thead><tr><th align="center">知识点</th><th align="center">描述</th></tr></thead><tbody><tr><td align="center">super()函数</td><td align="center">用于初始化继承自nn.Module的参数，实现子类与父类方法的关联。</td></tr><tr><td align="center">模型保存与加载</td><td align="center">支持整个网络加参数和仅参数两种保存形式，可以使用.pkl或.pth文件。</td></tr><tr><td align="center">卷积相关</td><td align="center">包括卷积核参数共享、局部连接、深度可分离卷积等概念。</td></tr><tr><td align="center">DataLoader</td><td align="center">用于数据加载，支持批量处理、随机打乱、自定义样本处理等。</td></tr><tr><td align="center">初始化方式</td><td align="center">卷积层和全连接层权重采用He-Uniform初始化，bias采用（-1，1）均匀分布。</td></tr></tbody></table></div>



应用场景：
研究环境中，尤其是需要反复修改模型结构的实验场景。
计算机视觉、自然语言处理等领域。
核心组件：

torch：核心库，包含张量操作、数学函数等。
torch.nn：神经网络模块，提供卷积层、全连接层等。
torch.optim：优化器模块，提供SGD、Adam等优化算法。 



### 1.  

##   训练模型

在每批训练开始时初始化梯度
前向传播
反向传播
计算损失并更新权重
```python

# 在数据集上循环多次
for epoch in range(2):  
    for i, data in enumerate(trainloader, 0):
        # 获取输入; data是列表[inputs, labels]
        inputs, labels = data 
        # (1) 初始化梯度
        optimizer.zero_grad() 

        # (2) 前向传播
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # (3) 反向传播
        loss.backward()
        # (4) 计算损失并更新权重
        optimizer.step()


```

(4)控制CPU与GPU模式的比较
对于Pytorch，你必须显式地为每个torch张量和numpy变量启用GPU。这将使代码变得混乱，如果你在CPU和GPU之间来回移动以执行不同的操作，则很容易出错。
```python
#获取GPU设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#传送网络到GPU
net.to(device)

# 传送输入和标签到GPU
inputs, labels = data[0].to(device), data[1].to(device)

JAVASCRIPT 复制 全屏

```

# 卷积
```python
#pytorch
self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)

```

# 反卷积
```python
#pytorch
self.dconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, bias=bias)
```
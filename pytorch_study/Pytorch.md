
Pytorch
Pytorch是一个深度学习框架(类似于TensorFlow)，由Facebook的人工智能研究小组开发。与Keras一样，它也抽象出了深层网络编程的许多混乱部分。

就高级和低级代码风格而言，Pytorch介于Keras和TensorFlow之间。比起Keras具有更大的灵活性和控制能力，但同时又不必进行任何复杂的声明式编程(declarative programming)。


## 特性

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
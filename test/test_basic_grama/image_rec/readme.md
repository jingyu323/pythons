# 图像识别
## EasyOCR 文字识别
## 简介
EasyOCR 是一个用于从图像中提取文本的 python 模块, 它是一种通用的 OCR，既可以读取自然场景文本，也可以读取文档中的密集文本。
利用cpu 识别文本

安装  EasyOCR 后会有 cv2.imread 不能使用的问题
解决方法：
1.先卸载opencv

pip uninstall opencv-python-headless
pip uninstall opencv-python
2.再下载opencv

pip install opencv-python

## 去除噪声

### 双边滤波
def bilateralFilter(src, d, sigmaColor, sigmaSpace, dst=None, borderType=None)
``` 
src: 输入的原始图像。
d: 表示在过滤过程中每个像素邻域的直径范围。如果这个值是非正数，则函数会从第五个参数sigmaSpace计算该值。
sigmaColor:颜色空间过滤器的sigma值，这个参数的值月大，表明该像素邻域内有越宽广的颜色会被混合到一起，产生较大的半相等颜色区域。 （这个参数可以理解为值域核的）
sigmaSpace:，如果该值较大，则意味着颜⾊相近的较远的像素将相互影响，从而使更⼤的区域中足够相似的颜色获取相同的颜色。当d>0时，d指定了邻域大小，那么不考虑sigmaSpace值，否则d正比于sigmaSpace。（这个参数可以理解为空间域核的）
dst:输出图像。
borderType:用于推断图像外部像素的某种边界模式，有默认值BORDER_DEFAULT。 
```
双边滤波器可以很好的保存图像边缘细节而滤除掉低频分量的噪音，但是双边滤波器的效率不是太高，花费的时间相较于其他滤波器而言也比较长。
对于简单的滤波而言，可以将两个sigma值设置成相同的值，如果值<10，则对滤波器影响很小，如果值>150则会对滤波器产生较大的影
## 图像平滑处理 去除噪点
###  均值滤波
对核中的取平均值
### 高斯滤波
###  中值滤波
去除噪点比高斯滤波效果好
##  边缘检测
###  canny  边缘检测

### 形态学
#### 腐蚀  
####  膨胀
#### 开运算 
#### sobel 算子
查找梯度 检测边缘

#### 礼帽  黑帽
礼帽：
原始输入-开运算结果

黑帽：闭运算- 原始输入
去除图像外较小明亮的噪点

#### 闭运算
去除图像内部较小明亮的噪点

### 

## 文字识别库
### 光学字符识别 pytesseract  
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  pytesseract
 pip install flask  -i https://pypi.tuna.tsinghua.edu.cn/simple 
 pip install pytesseract  -i https://pypi.tuna.tsinghua.edu.cn/simple 
 pip install easyocr  -i https://pypi.tuna.tsinghua.edu.cn/simple 
 pip install opencv-python  -i https://pypi.tuna.tsinghua.edu.cn/simple 
 pip install keras  -i https://pypi.tuna.tsinghua.edu.cn/simple 
 pip install openpyxl  -i https://pypi.tuna.tsinghua.edu.cn/simple 
 pip install opencv-contrib-python  -i https://pypi.tuna.tsinghua.edu.cn/simple 




可以查看支持那些语言
https://www.jaided.ai/easyocr/


# open-CV2

# 学习地址
## 各种识别方法
https://geek-docs.com/opencv/python-opencv/t_how-to-check-if-an-image-contour-is-convex-or-not-in-opencv-python.html
keras
https://keras-zh.readthedocs.io/applications/#vgg16

## 图像识别

- easyocr  对数字  中文 识别好 在gpu 下效率 高  输出小写
- Tesseract 对字母 识别好 在 cpu 下效率高 大写 

- opencv 用于图像识别 
- pytorch  深度学习
###  图像拼接

# 机器学习

## 特征数据提取

## 特征工程

### 图片拼接
步骤：
1. 读图片
2. 灰度化处理
3. 计算各自的特征点和描述子
4. 匹配特征.
5. 根据匹配到的特征，计算单应性矩阵
6. 对图片进行透视变换
7. 创建一个大图.
8. 放入两张图

## 算法分类
### 监督学习 -有特征值 有目标值
目标连续-- 回归算法
目标离散--分类

### 半监督学习  - 一部分有目标值 一部分没有

### 无监督学习 仅有特征值
###  强化学习
马尔科夫决策 动态规划

## 模型评估

- 准确吕
- 精确率
- 召回率
- F1-score
- AUC指标

### 回归模型评估

- 均方根误差
- 相对平方根误差
- 平均绝对值误差


## matplotlib 
主要用于2D图绘制 


## 待做
### 停车场
 主要学习如何定位指定区域
    - 根据坐标点用mask 反选
 
### 答题卡
- 如何识别选中的   
  - 根据二值化之后对选中的颜色进行阈值选中 检测器中的像素点的多少
- 如何指出来错误的 
  - 根据 识别出来正确的就识别出来错误的 再把正确的答案的位置画上轮廓

ln -s  /Library/Frameworks/Python.framework/Versions/3.11/bin/pip /usr/local/bin/pip


path = " ~/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5"


# 关键命令

## 绘制轮廓
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
- mode：轮廓检索模式，可以是以下值之一：
  - cv2.RETR_EXTERNAL：只检索最外层的轮廓。 
  - cv2.RETR_LIST：检索所有轮廓，并保存到列表中。 
  - cv2.RETR_CCOMP：检索所有轮廓，并组织成两层结构。外层轮廓和内层轮廓（如果存在的话）。 
  - cv2.RETR_TREE：检索所有轮廓，并组织成层次结构。
- method：轮廓近似方法，可以是以下值之一： 
  - cv2.CHAIN_APPROX_NONE：存储所有轮廓点的信息。 
  - cv2.CHAIN_APPROX_SIMPLE：仅存储轮廓的端点。

图像切割： 开始坐标 x+宽 y+高
 image[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
 cv2.imwrite(str(i)+'.jpg', splite_image)

1. 找到匹配程度最小的位置，即最佳匹配位置
```python
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
2. 在源图像上绘制匹配结果
top_left = max_loc
bottom_right = (top_left[0] + templ.shape[1], top_left[1] + templ.shape[0])
cv2.rectangle(image, top_left, bottom_right, 255, 2) 
```

##  光流估计

dib ssd 


## keras

### 模型训练步骤
#### equential模型如下 
1.   定义模型
```
from keras.models import Sequential

model = Sequential()
```
2.  添加网络层
```python
model.add(Dense(units=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(units=10))
model.add(Activation("softmax"))
```
3. .compile()方法来编译模型
```python
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
```
自己定制损失函数

```python
from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
```

4. 训练网络
```python

model.fit(x_train, y_train, epochs=5, batch_size=32)

```

单批次训练
```python
model.train_on_batch(x_batch, y_batch)
```
5. 模型进行评估
```python

oss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```
6. 进行预测
```python
classes = model.predict(x_test, batch_size=128)
```
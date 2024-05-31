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

可以查看支持那些语言
https://www.jaided.ai/easyocr/


# open-CV2

# 学习地址
## 各种识别方法
https://geek-docs.com/opencv/python-opencv/t_how-to-check-if-an-image-contour-is-convex-or-not-in-opencv-python.html
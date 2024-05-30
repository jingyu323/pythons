# EasyOCR
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


## 识别库
### 光学字符识别 pytesseract  
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  pytesseract


# open-CV2

# 学习地址
## 各种识别方法
https://geek-docs.com/opencv/python-opencv/t_how-to-check-if-an-image-contour-is-convex-or-not-in-opencv-python.html
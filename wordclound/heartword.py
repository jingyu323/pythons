#coding:utf-8

import os
from scipy.misc import imread
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
import matplotlib.pylab as plt
from os import path


#   解析图片
back_color = imread("./image/xin.jpg")

font = "C:\Windows\Fonts\SIMYOU.TTF"
wc = WordCloud(background_color="white",    #   背景颜色
               max_words=500,              #   最大词数
               mask=back_color,             #   掩膜，产生词云背景的区域，以该参数值作图绘制词云，这个参数不为空时，width,height会被忽略
               max_font_size=80,           #   显示字体的最大值
               stopwords=STOPWORDS.add("差评"),   #   使用内置的屏蔽词，再添加一个
               font_path=font,              #   解决显示口字型乱码问题，可进入C:/Windows/Fonts/目录更换字体
               random_state=42,             #   为每一词返回一个PIL颜色
               prefer_horizontal=10)        #   调整词云中字体水平和垂直的多少
# get data directory (using getcwd() is needed to support running example in generated IPython notebook)
ed = path.dirname(__file__) if "__file__" in locals() else os.getcwd()


# Read the whole text.
text = open(path.join(ed,'resource','xin.txt'),'r',encoding='UTF-8').read()

wordcloud =wc.generate(text)

#   从背景图片生成颜色值
image_colors = ImageColorGenerator(back_color)
# 显示图片
plt.imshow(wc)
# 关闭坐标轴
plt.axis("off")
# 绘制词云
plt.figure() #matalab 画出来的


#重新绘制字体颜色
plt.imshow(wc.recolor(color_func=image_colors))
plt.axis("off")
plt.show() #用来将图像显示在控制台
# 保存图片
wc.to_file("text2.png")


#coding:utf-8
import os

from os import path
from wordcloud import WordCloud

#使用os.path.join第二个参数的首个字符如果是"/" , 拼接出来的路径会不包含第一个参数。。。
'''
汉字乱码是因为字体不支持
'''

# get data directory (using getcwd() is needed to support running example in generated IPython notebook)
ed = path.dirname(__file__) if "__file__" in locals() else os.getcwd()


# Read the whole text.
text = open(path.join(ed,'resource','constitution.txt'),'r',encoding='UTF-8').read()

# Generate a word cloud image
wordcloud = WordCloud(font_path='simfang.ttf').generate(text)

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

# lower max_font_size
wordcloud = WordCloud(font_path='simfang.ttf',max_font_size=40).generate(text)
plt.figure() #不用这个方法也能显示
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# The pil way (if you don't have matplotlib)
# image = wordcloud.to_image()
# image.show()
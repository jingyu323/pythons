# coding : UTF-8
"""
requests：用来抓取网页的html源代码
csv：将数据写入到csv文件中
random：取随机数
time：时间相关操作
socket和http.client 在这里只用于异常处理
BeautifulSoup：用来代替正则式取源码中相应标签中的内容
urllib.request：另一种抓取网页的html源代码的方法，但是没requests方便（我一开始用的是这一种）
"""
import urllib

import requests
import csv
import random
import time
import socket
import http.client
from bs4 import BeautifulSoup

response = urllib.request.urlopen("http://www.163.com")
print( response.read())
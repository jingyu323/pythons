# -*- coding: utf-8 -*-
import requests        #导入requests包
url = 'http://www.cntour.cn/'
strhtml = requests.get(url)        #Get方式获取网页数据
print(strhtml.text)

from urllib import parse

import urllib.request
# urlopen()向URL发请求,返回响应对象,注意url必须完整
response=urllib.request.urlopen('http://www.baidu.com/')
print(response)


query_string = {
'wd' : '爬虫'
}
#调用parse模块的urlencode()进行编码
result = parse.urlencode(query_string)
#使用format函数格式化字符串，拼接url地址
url = 'http://www.baidu.com/s?{}'.format(result)
print(url)

from urllib import request
from urllib import parse
#重构请求头
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:6.0) Gecko/20100101 Firefox/6.0'}
#创建请求对应
req = request.Request(url=url,headers=headers)
#获取响应对象
res = request.urlopen(req)
#获取响应内容
html = res.read().decode("utf-8")

print(html)
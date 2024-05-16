import bisect
import csv
import http
import random
from datetime import time
from socket import socket

import requests
from bs4 import BeautifulSoup

list=[1,2,3,4]

tem = list


tem[0]=3

print(list)


dp = [1] * len(list)

print(dp)

# index = bisect.bisect_left(res, arr[i])

import numpy as np
mylist = [1,2,3]
print(tuple(mylist))
iarray = np.array(tuple(mylist))
print( iarray)

names = ['a', 'b', 'c', 'd', 'b']
names.remove('b')

import numpy as np

# 将列表转换为numpy的数组
a = np.array(["a", "b", "c", "a", "d", "a"])
# 获取元素的下标位置
eq_letter = np.where(a == "a")
print(eq_letter[0])  # [0 3 5]


nums = [1,5,7,8,9,6,3,11,20,17]
n, sum = len(nums), sum(nums)
isOK = [[False]*(sum//2+1) for _ in range(n//2+1)]
isOK[0][0]=True
for k in range(1,n+1):
    for i in range(min(k,n//2),0,-1):
        for v in range(1,sum//2+1):
            if v >= nums[k-1] and isOK[i-1][v-nums[k-1]]:
                isOK[i][v] = True
for i in range(n//2+1):
    for j in range(sum//2+1):
        print(isOK[i][j],end=' ')



#将列表转换为numpy的数组
a = np.array([["a","b","c"],["a","d","a"]])
#获取元素的下标位置
eq_letter = np.argwhere(a == "a")
print(eq_letter)



def get_content(url , data = None):
    header={
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, sdch',
        'Accept-Language': 'zh-CN,zh;q=0.8',
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.235'
    }
    timeout = random.choice(range(80, 180))
    while True:
        try:
            rep = requests.get(url,headers = header,timeout = timeout)
            rep.encoding = 'utf-8'
            # req = urllib.request.Request(url, data, header)
            # response = urllib.request.urlopen(req, timeout=timeout)
            # html1 = response.read().decode('UTF-8', errors='ignore')
            # response.close()
            break
        # except urllib.request.HTTPError as e:
        #         print( '1:', e)
        #         time.sleep(random.choice(range(5, 10)))
        #
        # except urllib.request.URLError as e:
        #     print( '2:', e)
        #     time.sleep(random.choice(range(5, 10)))
        except socket.timeout as e:
            print( '3:', e)
            time.sleep(random.choice(range(8,15)))

        except socket.error as e:
            print( '4:', e)
            time.sleep(random.choice(range(20, 60)))

        except http.client.BadStatusLine as e:
            print( '5:', e)
            time.sleep(random.choice(range(30, 80)))

        except http.client.IncompleteRead as e:
            print( '6:', e)
            time.sleep(random.choice(range(5, 15)))

    return rep.text
    # return html_text


def get_data(html_text):
    final = []
    bs = BeautifulSoup(html_text, "html.parser")  # 创建BeautifulSoup对象
    body = bs.body # 获取body部分
    data = body.find('div', {'id': '7d'})  # 找到id为7d的div
    ul = data.find('ul')  # 获取ul部分
    li = ul.find_all('li')  # 获取所有的li

    for day in li: # 对每个li标签中的内容进行遍历
        temp = []
        date = day.find('h1').string  # 找到日期
        temp.append(date)  # 添加到temp中
        inf = day.find_all('p')  # 找到li中的所有p标签
        temp.append(inf[0].string,)  # 第一个p标签中的内容（天气状况）加到temp中
        if inf[1].find('span') is None:
            temperature_highest = None # 天气预报可能没有当天的最高气温（到了傍晚，就是这样），需要加个判断语句,来输出最低气温
        else:
            temperature_highest = inf[1].find('span').string  # 找到最高温
            temperature_highest = temperature_highest.replace('℃', '')  # 到了晚上网站会变，最高温度后面也有个℃
        temperature_lowest = inf[1].find('i').string  # 找到最低温
        temperature_lowest = temperature_lowest.replace('℃', '')  # 最低温度后面有个℃，去掉这个符号
        temp.append(temperature_highest)   # 将最高温添加到temp中
        temp.append(temperature_lowest)   #将最低温添加到temp中
        final.append(temp)   #将temp加到final中

    return final

def write_data(data, name):
    file_name = name
    with open(file_name, 'a', errors='ignore', newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerows(data)



if __name__ == '__main__':
    url ='http://www.weather.com.cn/weather/101190401.shtml'
    html = get_content(url)
    result = get_data(html)
    write_data(result, 'weather.csv')

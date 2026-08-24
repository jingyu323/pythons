import requests
from lxml import etree
import csv

# 请求URL
url = 'http://www.weather.com.cn/weather1d/101010100.shtml'
# 请求头部
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
}

# 解析页面函数
def parse_html(html):
    selector = etree.HTML(html)
    city = selector.xpath('//*[@id="around"]/div/div[1]/div[1]/h1/text()')[0]
    temperature = selector.xpath('//*[@id="around"]/div/div[1]/div[1]/p/i/text()')[0]
    weather = selector.xpath('//*[@id="around"]/div/div[1]/div[1]/p/@title')[0]
    wind = selector.xpath('//*[@id="around"]/div/div[1]/div[1]/p/span/text()')[0]
    return city, temperature, weather, wind

# 保存数据函数
def save_data():
    f = open('beijing_weather.csv', 'w', newline='', encoding='utf-8-sig')
    writer = csv.writer(f)
    writer.writerow(['城市', '温度', '天气', '风力'])
    for i in range(10):
        response = requests.get(url, headers=headers)
        city, temperature, weather, wind = parse_html(response.text)
        writer.writerow([city, temperature, weather, wind])
    f.close()

if __name__ == '__main__':
    save_data()

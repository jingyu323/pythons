from datetime import time

import lxml
from bs4 import BeautifulSoup
from selenium import webdriver

from ChromeDriverProvider import ChromeDriverProvider

# 指定ChromeDriver路径


url = f'https://api3.cls.cn/share/quote/analysis'
chromeDriverProvider = ChromeDriverProvider(url=url)
html =chromeDriverProvider.getHtmlSource(url)

print(html)
soup = BeautifulSoup(html, lxml)
print(soup)




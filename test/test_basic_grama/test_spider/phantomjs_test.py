from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')#上面三行代码就是为了将Chrome不弹出界面，实现无界面爬取
browser = webdriver.Chrome(chrome_options=chrome_options)

# 如果没有在环境变量指定Phantomjs位置# driver = webdriver.Phantomjs(executable_path="./phantomjs"))
# get方法会一直等到页面被完全加载，然后才会继续程序，通常测试会在这里选择 time.sleep(2)
browser.get("https://www.taobao.com")
# 获取页面名为 wrapper的id标签的文本内容
print(browser.page_source)
# 关闭浏览器

input_first = browser.find_element(By.ID,'q')

input_second = browser.find_element(By.CLASS_NAME,'.q')



print(input_first,input_second)


browser.close()
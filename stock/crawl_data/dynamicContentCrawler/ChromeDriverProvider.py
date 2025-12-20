from bs4 import BeautifulSoup

from selenium import webdriver

from selenium.webdriver.chrome.options import Options

from selenium.webdriver.chrome.service import Service


class  ChromeDriverProvider:
    def __init__(self,url):
        self.url = url

    def getHtmlSoup(self):
        # 设置Chrome选项
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # 不打开浏览器窗口
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")

        # 指定ChromeDriver路径
        service = Service(
            executable_path=r'C:/Program Files (x86)/Google/Chrome/Application/chromedriver.exe')  # 替换为你的chromedriver路径

        # 初始化WebDriver
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get(self.url)
        # time.sleep(5)  # 等待页面加载完成，可以根据实际情况调整等待时间

        # 获取并解析HTML结构
        page_source = driver.page_source
        print(page_source)
        soup = BeautifulSoup(page_source, 'html.parser')

        # 关闭浏览器
        driver.quit()
        return soup

    def getHtmlSource(self,url):
        # 设置Chrome选项
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # 不打开浏览器窗口
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")

        # 指定ChromeDriver路径
        service = Service(executable_path=r'C:/Program Files (x86)/Google/Chrome/Application/chromedriver.exe')  # 替换为你的chromedriver路径
        # 初始化WebDriver
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get(url)
        # time.sleep(5)  # 等待页面加载完成，可以根据实际情况调整等待时间

        # 获取并解析HTML结构
        page_source = driver.page_source

        # 关闭浏览器
        driver.quit()
        return page_source
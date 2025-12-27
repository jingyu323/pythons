import asyncio
from pyppeteer import launch
from pyquery import PyQuery as pq
"""


"""

class PyppeteerBrowser:
      def __init__(self,url):
         self.url = url


async def main(self):
    # 创建浏览器对象
    # browser = await launch(headless=False, args=['--disable-infobars'])

    browser = await launch({
        'headless': False,
        'executablePath': 'D:/soft/chrome-win/chrome.exe'
    },args=['--disable-infobars'])

    # 打开新的标签页
    page = await browser.newPage()

    # 设置视图大小
    await page.setViewport({'width': 1366, 'height': 768})

    # 设置UserAgent
    await page.setUserAgent(
        'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36')

    # 访问页面
    response = await page.goto( self.url)

    # 获取status、headers、url
    # print(response.status)
    # print(response.headers)
    await page.waitForSelector('.a-plate-stock-list ')
    doc = pq(await page.content())

    await browser.close()
    return doc


asyncio.get_event_loop().run_until_complete(main())


if __name__ == '__main__':
    pb = PyppeteerBrowser(url='https://api3.cls.cn/share/quote/analysis')
    print(pb)

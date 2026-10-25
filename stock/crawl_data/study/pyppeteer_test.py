import asyncio
from pyppeteer import launch

width, height = 1366, 768
#  设置无头模式不再会弹出浏览器界面
async def main():
    # browser = await launch()

    browser = await launch({
        'headless': True,
        # 'devtools': True,
        'executablePath': 'D:/soft/chrome-win/chrome.exe'
    },args=['--disable-infobars'])

    page = await browser.newPage()
    await page.setViewport({'width': width, 'height': height})
    await page.goto('https://spa2.scrape.center/')
    await page.waitForSelector('.item .name')
    await asyncio.sleep(2)
    await page.screenshot({'path': 'screen.png'})
    dimensions = await page.evaluate("""() => {
    return {
        width: document.documentElement.clientWidth,
        height: document.documentElement.clientHeight,
        deviceScaleFactor: window.devicePixelRatio,
    }
    }""")
    print(dimensions)
# 执行点击事件
    await page.waitForSelector('.item .name')
    await page.click('.item .name', options={
        'button': 'left',
        'clickCount': 1,  # 1 或 2
        'delay': 3000  # 毫秒
    })

    await browser.close()
asyncio.get_event_loop().run_until_complete(main())
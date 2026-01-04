import json

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# 代理配置
proxyHost = "www.16yun.cn"
proxyPort = "5445"
proxyUser = "16QMSOML"
proxyPass = "280651"

def parse_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    # 例如，获取所有的段落标签
    paragraphs = soup.find_all('p')
    # for p in paragraphs:
    #     print(p.text)

def main():
    with sync_playwright() as p:
        # 启动Chromium浏览器并配置代理
        browser = p.chromium.launch(
            headless=True,  # 无头模式
            # proxy={
            #     "server": f"http://{proxyHost}:{proxyPort}",
            #     "username": proxyUser,
            #     "password": proxyPass,
            # }
        )

        # 创建新页面
        page = browser.new_page()
        stock_list = []
        try:
            page.goto("https://api3.cls.cn/share/quote/analysis", timeout=10000)  # 增加超时设置
            html_content = page.content()

            parse_html(html_content)

            # 提取数据
            element = page.query_selector("section.a-plate-stock-list")
            # print(element)
            steles = element.query_selector_all("div.a-plate-stock-wrap")

            for stele in steles:
                # print(stele.inner_text())
                catgorys = stele.query_selector("div.a-plate-stock-box")
                catege = catgorys.query_selector("div.c-6b4122").text_content()
                # print("分类：", catege)
                stocks = stele.query_selector_all("div.a-stock-box")
                for stock in stocks:
                    stock_name = stock.query_selector("div.a-stock-name").text_content()
                    stock_code = stock.query_selector("div.a-stock-code").text_content()
                    stock_last = stock.query_selector("div.a-stock-last").text_content()
                    stock_change = stock.query_selector("div.a-stock-change").text_content()
                    stock_time = stock.query_selector("div.a-stock-time").text_content()
                    stock_cmc = stock.query_selector("div.a-stock-cmc").text_content()
                    stock_reason = stock.query_selector("div.a-stock-reason").text_content()
                    stock_tag = stock.query_selector("span.a-stock-tag")
                    stock_tag_txt =""
                    if stock_tag :
                        # print("stock_tag:", stock_tag.inner_text())
                        stock_tag_txt = stock_tag.inner_text()

                    # print("stock_name:", stock_name)
                    # print("stock_code:", stock_code)
                    # print("stock_last:", stock_last)
                    # print("stock_change:", stock_change)
                    # print("stock_time:", stock_time)
                    # print("stock_cmc:", stock_cmc)
                    # print("stock_reason text_content :", stock_reason)
                    #
                    # print("==========================")

                    stock_data = {
                        'stockCode': stock_code.strip(),
                        'stockName': stock_name.strip(),
                        'endPrice': stock_last.strip(),
                        'topic': catege.strip(),
                        'dealMount': stock_cmc.strip(),
                        'analysis': stock_reason.strip(),
                        'stock_tag': stock_tag_txt.strip()
                    }
                    stock_list.append(stock_data)


            browser.close()
        except Exception as e:
            print(f"发生错误: {e}")

        finally:
            # 确保浏览器关闭
            browser.close()

        return stock_list


if __name__ == "__main__":
    stock_list =  main()
    json_str = json.dumps(stock_list)
    print(json_str)

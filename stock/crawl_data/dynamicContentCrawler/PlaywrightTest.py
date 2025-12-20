from playwright.sync_api import sync_playwright

# 代理配置
proxyHost = "www.16yun.cn"
proxyPort = "5445"
proxyUser = "16QMSOML"
proxyPass = "280651"

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

    try:
        # 访问网页并等待加载
        page.goto("https://api3.cls.cn/share/quote/analysis", timeout=10000)  # 增加超时设置


        # 获取渲染后的HTML
        rendered_html = page.content()
        print(rendered_html)

        # 提取数据
        element = page.query_selector("div.dynamic-content")
        if element:
            print(element.inner_text())
        else:
            print("目标元素未找到")

    except Exception as e:
        print(f"发生错误: {e}")

    finally:
        # 确保浏览器关闭
        browser.close()

html.parser


https://googlechromelabs.github.io/chrome-for-testing/

# 等待页面加载完成  https://developer.baidu.com/article/details/3270429
wait = WebDriverWait(browser, 10) 
element = wait.until(EC.presence_of_element_located((By.ID, 'some_id'))



| **方法**          | **适用场景**             | **速度** | **复杂度** |
| ----------------- | ------------------------ | -------- | ---------- |
| **Selenium**      | 需要兼容多种浏览器       | 慢       | 中等       |
| **Playwright**    | 高性能、现代浏览器自动化 | 快       | 低         |
| **Pyppeteer**     | 直接控制Chrome           | 快       | 中高       |


html.parse 解析不出来数据 用lxml 就能获取数据

python 中没有三目运算符 可以使用如下等式代替

max_value = a if a > b else b

# 爬虫

# #推荐（单次定位）
title = soup.select_one('head > title').text
# 限定范围（快）
content = soup.find(id='content')
content_links = content.find_all('a')

# 只解析产品区域
product_only = SoupStrainer('div', class_='product')
soup = BeautifulSoup(html, 'lxml', parse_only=product_only)

# 多次使用同一选择器
product_selector = 'div.product-item'
for page in pages:
    products = page.select(product_selector)


# (pyppeteer) 文档
https://miyakogi.github.io/pyppeteer/reference.html <br>
项目地址：https://github.com/miyakogi/pyppeteer<br>
官方文档：https://miyakogi.github.io/pyppeteer/reference.html<br>

# 升级
pip install --upgrade pandas

53c09ab9e1ab53bb37e4b7ddd32a027e8ac60a02e437cdec4ebff349



## 方案对比与选型建议

| 方案类型      | 优点                   | 缺点                   | 适用场景           |
| :------------ | :--------------------- | :--------------------- | :----------------- |
| 免费API       | 无需注册，快速上手     | 数据量有限，可能不稳定 | 快速验证、小型项目 |
| Tushare Pro   | 数据全面，专业性强     | 需要注册，有调用限制   | 量化研究、专业分析 |
| AKShare       | 开源免费，更新及时     | 需要处理反爬机制       | 学术研究、个人项目 |
| 网页爬取      | 数据源丰富，灵活可控   | 反爬风险高，稳定性差   | 特定数据源获取     |
| Yahoo Finance | 支持国际市场，接口简单 | 数据延迟，国际市场为主 | 跨国投资分析       |

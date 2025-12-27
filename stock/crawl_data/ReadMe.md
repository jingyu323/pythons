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
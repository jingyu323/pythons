import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}
url = 'https://api3.cls.cn/share/quote/analysis'
response = requests.get(url,headers=headers,timeout=15)

# 检查请求是否成功
if response.status_code == 200:
    html_content = response.text
    print(html_content)  # 打印HTML内容，或者进行下一步处理
else:
    print("Failed to retrieve the webpage")

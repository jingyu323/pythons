
import requests
from bs4 import BeautifulSoup
import time
import random
import csv
from urllib.parse import urljoin

def get_margin_trading_stocks():
    """
    爬取同花顺融资融券标的股票数据
    """
    # 设置请求头，模拟浏览器访问


    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

    # 同花顺融资融券标的页面URL
    base_url = "http://data.10jqka.com.cn/market/rzrq/"
    
    # 存储所有股票数据
    all_stocks = []
    
    # 创建会话
    session = requests.Session()
    session.headers.update(headers)

    
    try:
        # 发送请求获取第一页数据
        response = session.get(base_url)
        response.encoding = 'gbk'
        soup = BeautifulSoup(response.text, 'html.parser')
        print(soup)
        # 解析第一页数据
        parse_page_data(soup, all_stocks)
        
        # 查找分页信息
        pagination = soup.find('div', class_='m-page')
        if pagination:
            # 获取总页数
            page_links = pagination.find_all('a')
            max_page = 1
            for link in page_links:
                if link.text.isdigit():
                    page_num = int(link.text)
                    if page_num > max_page:
                        max_page = page_num
            
            # 遍历剩余页面
            for page in range(2, max_page + 1):
                print(f"正在爬取第 {page} 页...")
                page_url = f"http://data.10jqka.com.cn/market/rzrq/board/ALL/field/zdf/order/desc/page/{page}/"
                response = session.get(page_url)
                response.encoding = 'gbk'
                soup = BeautifulSoup(response.text, 'html.parser')
                parse_page_data(soup, all_stocks)
                
                # 添加随机延时，避免被反爬
                time.sleep(random.uniform(1, 3))
                
    except Exception as e:
        print(f"爬取过程中出现错误: {e}")
    
    return all_stocks

def parse_page_data(soup, stocks_list):
    """
    解析页面中的股票数据
    """
    # 查找包含股票信息的表格
    table = soup.find('table', class_='m-table m-pager-table')
    if not table:
        print("未找到数据表格")
        return
    
    # 查找表格主体
    tbody = table.find('tbody')
    if not tbody:
        tbody = table
    
    # 遍历每一行数据
    rows = tbody.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        if len(cols) >= 3:  # 确保有足够的列数据
            try:
                # 提取股票代码和名称
                code_col = cols[1]
                name_col = cols[2]
                
                stock_code = code_col.get_text(strip=True)
                stock_name = name_col.get_text(strip=True)
                
                # 提取其他信息（如需更多字段可扩展）
                # 这里可以根据需要添加更多字段的提取逻辑
                
                stocks_list.append({
                    'code': stock_code,
                    'name': stock_name
                })
            except Exception as e:
                print(f"解析行数据时出错: {e}")
                continue

def save_to_csv(stocks_data, filename='margin_trading_stocks.csv'):
    """
    将股票数据保存到CSV文件
    """
    if not stocks_data:
        print("没有数据可保存")
        return
    
    with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['code', 'name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for stock in stocks_data:
            writer.writerow(stock)
    
    print(f"数据已保存到 {filename}，共 {len(stocks_data)} 条记录")

def main():
    """
    主函数
    """
    print("开始爬取同花顺融资融券标的股票数据...")
    
    # 获取股票数据
    stocks = get_margin_trading_stocks()
    
    if stocks:
        print(f"成功获取 {len(stocks)} 只融资融券标的股票")
        
        # 显示前10只股票作为示例
        print("\n前10只股票示例:")
        for i, stock in enumerate(stocks[:10]):
            print(f"{i+1}. {stock['code']} - {stock['name']}")
        
        # 保存到CSV文件
        save_to_csv(stocks)
    else:
        print("未能获取到股票数据")

if __name__ == "__main__":
    main()

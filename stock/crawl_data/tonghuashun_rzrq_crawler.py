from operator import indexOf

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from datetime import datetime



class RZRQCrawler:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Referer': 'http://data.10jqka.com.cn/market/rzrq/',
            'Host': 'data.10jqka.com.cn'
        }
        self.base_url = "http://data.10jqka.com.cn/market/rzrq/"

    def crawl_rzrq_data(self):
        """爬取融资融券数据"""
        try:
            response = requests.get(self.base_url, headers=self.headers, timeout=10)
            response.encoding = 'gbk'

            if response.status_code == 200:
                # print(response.text)
                soup = BeautifulSoup(response.text, 'lxml')

                tdiv = soup.find('div', attrs={'class': 'page-table'})

                # 查找数据表格
                table = tdiv.find('table', attrs={'class': 'm-table'})

                # tables = tdiv.find_all('table')
                # for table in tables:
                #     print(table)

                # print(table)

                data_list = []
                if table:
                    tbody = table.find('tbody') # 跳过表头
                    rows = tbody.find_all('tr') # 跳过表头
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) > 0:
                            stock_data = {
                                '股票代码': cols[1].text.strip(),
                                '股票名称': cols[2].text.strip(),
                                '融资余额':   cols[3].text.strip().replace("亿","")  if  "亿" in cols[3].text.strip() else cols[3].text.strip().replace("万",""),
                                '融资买入额':  cols[4].text.strip().replace("亿","")  if  "亿" in cols[4].text.strip() else cols[4].text.strip().replace("万",""),
                                '融资偿还额':   cols[5].text.strip().replace("亿","")  if  "亿" in cols[5].text.strip() else cols[5].text.strip().replace("万",""),
                                '融资净买入额':  cols[6].text.strip().replace("亿","")  if  "亿" in cols[6].text.strip() else cols[6].text.strip().replace("万",""),
                                '融券余额': cols[7].text.strip() ,
                                '融券卖出量': cols[8].text.strip() ,
                                '融券净买入': cols[9].text.strip() ,
                                '融券净卖出': cols[10].text.strip() ,
                                '融资融券余额':  cols[11].text.strip().replace("亿","")  if  "亿" in cols[11].text.strip() else cols[11].text.strip().replace("万","")
                            }
                            data_list.append(stock_data)

                return pd.DataFrame(data_list)

        except Exception as e:
            print(f"爬取失败: {e}")
            return None

    def crawl_multiple_pages(self, max_pages=5):
        """分页爬取数据"""
        base_url = "http://data.10jqka.com.cn/market/rzrq/board/{}/field/rzrqye/order/desc/page/{}/"
        all_data = []

        for page in range(1, max_pages + 1):
            url = base_url.format("rzrq", page)
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.encoding = 'utf-8'

                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    table = soup.find('table', {'class': 'm-table'})

                    if table:
                        rows = table.find_all('tr')[1:]
                        for row in rows:
                            cols = row.find_all('td')
                            if len(cols) > 0:
                                stock_data = {
                                    '股票代码': cols[1].text.strip(),
                                    '股票名称': cols[2].text.strip(),
                                    '融资余额': cols[3].text.strip(),
                                    '融资买入额': cols[4].text.strip(),
                                    '融资偿还额': cols[5].text.strip(),
                                    '融资净买入额': cols[6].text.strip(),
                                    '融券余额': cols[7].text.strip(),
                                    '融券卖出量': cols[8].text.strip(),
                                    '融券净买入': cols[9].text.strip(),
                                    '融券净卖出': cols[10].text.strip(),
                                    '融资融券余额': cols[11].text.strip()
                                }

                                all_data.append(stock_data)

                time.sleep(1)  # 添加延时避免被封

            except Exception as e:
                print(f"第{page}页爬取失败: {e}")
                continue

        return pd.DataFrame(all_data)

    def clean_data(self, df):
        """数据清洗"""
        if df is None or df.empty:
            return df

        # 去除空值
        df = df.dropna()

        # 转换数据类型
        for col in ['融资余额', '融资买入额', '融券余额', '融券卖出量', '融资融券余额']:
            if col in df.columns:
                df[col] = df[col].str.replace(',', '').astype(float)

        return df

    def analyze_data(self, df):
        """数据分析"""
        if df is None or df.empty:
            return None
        # 融资余额排名前十
        top_10_rz = df.nlargest(10, '融资余额')
        print(top_10_rz)
        print("====")

        # 融券余额排名前十
        top_10_rq = df.nlargest(10, '融券余额')

        # 总余额统计
        total_balance = df['融资融券余额'].sum() if '融资融券余额' in df.columns else 0

        return {
            'total_balance': total_balance,
            'top_rz': top_10_rz,
            'top_rq': top_10_rq
        }

    def save_data(self, df, filename=None):
        """保存数据"""
        if df is None or df.empty:
            print("没有数据可保存")
            return

        if filename is None:
            filename = f"融资融券数据_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # 保存为CSV
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"数据已保存至: {filename}")

    def daily_crawl(self):
        """每日定时爬取"""
        print(f"{datetime.now()} 开始执行每日数据爬取")
        df = self.crawl_rzrq_data()
        if df is not None:
            df = self.clean_data(df)
            self.save_data(df)
            print("每日爬取任务完成")
        else:
            print("数据爬取失败")


def main():
    """主函数"""
    crawler = RZRQCrawler()

    print("🚀 开始爬取同花顺融资融券数据...")

    # 爬取数据
    df = crawler.crawl_rzrq_data()

    if df is not None:
        print(f"✅ 成功爬取 {len(df)} 条融资融券数据")
        print("\n前5条数据预览:")
        print(df.head())

        # 数据清洗
        df_cleaned = crawler.clean_data(df)

        # 数据分析
        analysis_result = crawler.analyze_data(df_cleaned)
        if analysis_result:
            print(f"\n📊 融资融券总余额: {analysis_result['total_balance']:,.2f} 元")
            print("\n📈 融资余额前十股票:")
            print(analysis_result['top_rz'][['股票名称', '融资余额']].to_string(index=False))
            print("\n📉 融券余额前十股票:")
            print(analysis_result['top_rq'][['股票名称', '融券余额']].to_string(index=False))

        # 保存数据
        crawler.save_data(df_cleaned)

        # 演示分页爬取功能
        print("\n🔄 演示分页爬取功能(前3页)...")
        df_multi = crawler.crawl_multiple_pages(max_pages=3)
        if not df_multi.empty:
            print(f"分页爬取共获得 {len(df_multi)} 条数据")
            crawler.save_data(df_multi, f"分页融资融券数据_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    else:
        print("❌ 数据爬取失败")


if __name__ == "__main__":
    main()

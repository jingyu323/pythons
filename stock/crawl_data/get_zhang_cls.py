
import requests
import json
import pandas as pd
from datetime import datetime
import time

from bs4 import BeautifulSoup


class StockAnalyzer:
    def __init__(self):
        self.base_url = "https://api3.cls.cn/share/quote/analysis"
        self.headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'
        }



    def fetch_stock_data(self):


        try:
            response = requests.get(self.base_url,  headers=self.headers, timeout=10)


            response.encoding = 'utf-8'

            if response.status_code == 200:
                print(response.text)
                print("========================================")
                print(response.content)
                soup = BeautifulSoup(response.text, 'html.parser')

                # 查找数据表格
                table = soup.find('section', {'class': 'a-plate-stock-list'})
                print(table)
                divs = table.find_all('div', {'class': 'a-plate-stock-wrap'})
                for div in divs:
                    print(div)

            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"数据获取失败: {e}")
            return None

    def parse_stock_data(self, data):
        """解析股票数据"""
        if not data:
            return None

        # 提取涨停股票信息
        limit_up_stocks = []

        # 解析各板块涨停股票
        sectors = ['核电', '智能驾驶', '商业航天', '房地产概念', 'IP经济',
                   '福建', '商业零售', '海南', '食品饮料', '纺织服装',
                   '光伏', '光通信', 'ST股']

        for sector in sectors:
            if sector in data:
                sector_data = data[sector]
                for stock in sector_data.get('stocks', []):
                    stock_info = {
                        '板块': sector,
                        '股票简称': stock.get('name', ''),
                        '股票代码': stock.get('code', ''),
                        '现价': stock.get('price', 0),
                        '涨幅': stock.get('increase', 0),
                        '涨停时间': stock.get('limit_up_time', ''),
                        '流通市值': stock.get('market_cap', ''),
                        '涨停天数': stock.get('limit_up_days', 1),
                        '概念说明': stock.get('concept', '')
                    }
                    limit_up_stocks.append(stock_info)

        return limit_up_stocks

    def analyze_by_sector(self, stocks_data):
        """按板块分析涨停股票"""
        sector_analysis = {}

        for stock in stocks_data:
            sector = stock['板块']
            if sector not in sector_analysis:
                sector_analysis[sector] = {
                    '涨停数量': 0,
                    '平均涨幅': 0,
                    '总流通市值': 0,
                    '股票列表': []
                }

            sector_analysis[sector]['涨停数量'] += 1
            sector_analysis[sector]['股票列表'].append(stock)

        return sector_analysis

    def generate_report(self, sector_analysis):
        """生成分析报告"""
        print("=" * 80)
        print(f"📈 涨停股票分析报告 - {datetime.now().strftime('%Y年%m月%d日')}")
        print("=" * 80)

        # 按涨停数量排序
        sorted_sectors = sorted(sector_analysis.items(),
                                key=lambda x: x[1]['涨停数量'], reverse=True)

        for sector, data in sorted_sectors:
            print(f"\n🔥 {sector}板块 (涨停{data['涨停数量']}只)")
            print("-" * 50)

            for stock in data['股票列表']:
                print(f"├─ {stock['股票简称']}({stock['股票代码']})")
                print(f"│  ├─ 现价: {stock['现价']}元 | 涨幅: {stock['涨幅']}%")
                print(f"│  ├─ 涨停时间: {stock['涨停时间']}")
                print(f"│  ├─ 流通市值: {stock['流通市值']}")
                if stock['概念说明']:
                    print(f"│  └─ 概念: {stock['概念说明']}")
                print(f"│")

        # 统计总览
        total_limit_up = sum(data['涨停数量'] for data in sector_analysis.values())
        print(f"\n📊 市场总览")
        print(f"├─ 总涨停股票: {total_limit_up}只")
        print(f"├─ 热门板块数量: {len(sector_analysis)}个")
        print(f"└─ 数据更新时间: {datetime.now().strftime('%H:%M:%S')}")

    def save_to_excel(self, stocks_data, filename=None):
        """保存数据到Excel文件"""
        if not filename:
            filename = f"涨停股票分析_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

        df = pd.DataFrame(stocks_data)
        df.to_excel(filename, index=False)
        print(f"\n💾 数据已保存到: {filename}")

    def main(self):
        """主函数"""
        print("🚀 开始获取股票数据...")

        # 获取数据
        raw_data = self.fetch_stock_data()
        print(raw_data)
        if not raw_data:
            return

        # 解析数据
        stocks_data = self.parse_stock_data(raw_data)
        if not stocks_data:
            print("❌ 数据解析失败")
            return

        print(f"✅ 成功获取 {len(stocks_data)} 只涨停股票数据")

        # 按板块分析
        sector_analysis = self.analyze_by_sector(stocks_data)

        # 生成报告
        self.generate_report(sector_analysis)

        # 保存数据
        self.save_to_excel(stocks_data)

        return stocks_data, sector_analysis

# 使用示例
if __name__ == "__main__":
    analyzer = StockAnalyzer()
    stocks_data, sector_analysis = analyzer.main()


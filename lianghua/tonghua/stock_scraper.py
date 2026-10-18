import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import json
import re
from datetime import datetime
import logging
from typing import Dict, List, Optional
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ThsStockScraper:
    """同花顺大涨股数据爬取类"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        })

    def get_rising_stocks(self, date: str = None) -> List[Dict]:
        """
        获取大涨股列表
        Args:
            date: 日期，格式为YYYYMMDD，默认为当天
        """
        if date is None:
            date = datetime.now().strftime('%Y%m%d')

        url = f"http://q.10jqka.com.cn/index/index/board/all/field/zdf/order/desc/page/1/ajax/1/"

        try:
            response = self.session.get(url)
            response.encoding = 'gb2312'
            soup = BeautifulSoup(response.text, 'html.parser')

            stocks = []
            table = soup.find('table', {'class': 'm-table m-pager-table'})

            if table:
                rows = table.find_all('tr')[1:]  # 跳过表头

                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 10:
                        stock_data = {
                            '股票代码': cols[1].text.strip(),
                            '股票名称': cols[2].text.strip(),
                            '最新价': cols[3].text.strip(),
                            '涨跌幅': cols[4].text.strip(),
                            '涨跌额': cols[5].text.strip(),
                            '成交量(手)': cols[6].text.strip(),
                            '成交额(万)': cols[7].text.strip(),
                            '振幅': cols[8].text.strip(),
                            '最高': cols[9].text.strip(),
                            '最低': cols[10].text.strip(),
                            '日期': date
                        }
                        # 获取更多详细信息
                        stock_code = stock_data['股票代码'][2:]  # 去掉市场前缀
                        detailed_info = self.get_stock_detail(stock_code)
                        if detailed_info:
                            stock_data.update(detailed_info)

                        stocks.append(stock_data)
                        logger.info(f"已获取股票: {stock_data['股票名称']}({stock_data['股票代码']})")

            return stocks

        except Exception as e:
            logger.error(f"获取大涨股列表失败: {e}")
            return []

    def get_stock_detail(self, stock_code: str) -> Optional[Dict]:
        """获取股票详细信息"""
        details = {}

        try:
            # 获取业绩信息
            performance = self.get_performance(stock_code)
            if performance:
                details.update(performance)

            # 获取涨停原因
            limit_up_reason = self.get_limit_up_reason(stock_code)
            if limit_up_reason:
                details['涨停原因分析'] = limit_up_reason

            # 获取融资融券余额
            margin_data = self.get_margin_data(stock_code)
            if margin_data:
                details.update(margin_data)

            # 获取当日买卖量
            trade_data = self.get_trade_data(stock_code)
            if trade_data:
                details.update(trade_data)

            # 获取散户数量（通过股东人数变化）
            retail_data = self.get_retail_investor_data(stock_code)
            if retail_data:
                details.update(retail_data)

            return details

        except Exception as e:
            logger.error(f"获取股票{stock_code}详细信息失败: {e}")
            return None

    def get_performance(self, stock_code: str) -> Optional[Dict]:
        """获取业绩信息"""
        try:
            url = f"http://basic.10jqka.com.cn/{stock_code}/flash.html"
            response = self.session.get(url)
            response.encoding = 'gb2312'

            # 这里需要根据实际网页结构解析业绩数据
            # 示例解析，实际需要调整选择器
            soup = BeautifulSoup(response.text, 'html.parser')

            performance = {}
            # 查找业绩相关的元素
            performance_divs = soup.find_all('div', {'class': 'm-box'})

            for div in performance_divs:
                text = div.text
                if '每股收益' in text:
                    performance['每股收益'] = re.search(r'每股收益\s*([\d.-]+)', text).group(1) if re.search(
                        r'每股收益\s*([\d.-]+)', text) else ''
                if '净利润' in text:
                    performance['净利润'] = re.search(r'净利润\s*([\d.-]+亿?)', text).group(1) if re.search(
                        r'净利润\s*([\d.-]+亿?)', text) else ''
                if '营业收入' in text:
                    performance['营业收入'] = re.search(r'营业收入\s*([\d.-]+亿?)', text).group(1) if re.search(
                        r'营业收入\s*([\d.-]+亿?)', text) else ''

            return performance if performance else None

        except Exception as e:
            logger.warning(f"获取业绩信息失败: {e}")
            return None

    def get_limit_up_reason(self, stock_code: str) -> Optional[str]:
        """获取涨停原因分析"""
        try:
            url = f"http://stockpage.10jqka.com.cn/{stock_code}/limitUp/"
            response = self.session.get(url)
            response.encoding = 'utf-8'

            soup = BeautifulSoup(response.text, 'html.parser')

            # 查找涨停原因
            reason_div = soup.find('div', {'class': 'limitup-reason'})
            if reason_div:
                return reason_div.text.strip()

            # 尝试其他可能的元素
            reason_elements = soup.find_all(['p', 'div', 'span'], text=re.compile(r'涨停|原因|概念'))
            for element in reason_elements:
                if len(element.text.strip()) > 10:  # 确保有一定长度
                    return element.text.strip()

            return "未找到涨停原因"

        except Exception as e:
            logger.warning(f"获取涨停原因失败: {e}")
            return None

    def get_margin_data(self, stock_code: str) -> Optional[Dict]:
        """获取融资融券余额"""
        try:
            url = f"http://data.10jqka.com.cn/market/rzrq/code/{stock_code}/"
            response = self.session.get(url)
            response.encoding = 'gb2312'

            soup = BeautifulSoup(response.text, 'html.parser')

            margin_data = {}
            table = soup.find('table', {'class': 'm-table'})

            if table:
                rows = table.find_all('tr')
                if len(rows) > 1:
                    latest_data = rows[1].find_all('td')
                    if len(latest_data) >= 4:
                        margin_data = {
                            '融资余额(亿)': latest_data[1].text.strip(),
                            '融券余额(亿)': latest_data[2].text.strip(),
                            '融资融券余额(亿)': latest_data[3].text.strip(),
                            '融资融券日期': latest_data[0].text.strip()
                        }

            return margin_data if margin_data else None

        except Exception as e:
            logger.warning(f"获取融资融券数据失败: {e}")
            return None

    def get_trade_data(self, stock_code: str) -> Optional[Dict]:
        """获取当日买卖量数据"""
        try:
            url = f"http://market.10jqka.com.cn/realhead/hs_{stock_code}.html"
            response = self.session.get(url)

            # 解析JSON数据
            data = response.json()

            trade_data = {}
            if 'items' in data:
                items = data['items']
                trade_data = {
                    '当日买入量(手)': items.get('buy', ''),
                    '当日卖出量(手)': items.get('sell', ''),
                    '净买入量(手)': items.get('net_buy', ''),
                    '大单净流入(万)': items.get('large_net_inflow', ''),
                    '中单净流入(万)': items.get('medium_net_inflow', ''),
                    '小单净流入(万)': items.get('small_net_inflow', '')
                }

            return trade_data if trade_data else None

        except Exception as e:
            logger.warning(f"获取交易数据失败: {e}")
            return None

    def get_retail_investor_data(self, stock_code: str) -> Optional[Dict]:
        """获取散户数量（通过股东人数变化）"""
        try:
            url = f"http://basic.10jqka.com.cn/{stock_code}/holder.html"
            response = self.session.get(url)
            response.encoding = 'gb2312'

            soup = BeautifulSoup(response.text, 'html.parser')

            retail_data = {}

            # 查找股东人数相关数据
            holder_divs = soup.find_all('div', {'class': 'm-table'})

            for div in holder_divs:
                text = div.text
                if '股东人数' in text:
                    # 提取最新一期股东人数
                    match = re.search(r'股东人数.*?(\d+\.?\d*[万千]?)', text)
                    if match:
                        retail_data['股东人数'] = match.group(1)

                if '人均流通股' in text:
                    match = re.search(r'人均流通股.*?(\d+\.?\d*)', text)
                    if match:
                        retail_data['人均流通股(股)'] = match.group(1)

            return retail_data if retail_data else None

        except Exception as e:
            logger.warning(f"获取散户数据失败: {e}")
            return None

    def save_to_excel(self, stocks_data: List[Dict], filename: str = None):
        """保存数据到Excel文件"""
        if not stocks_data:
            logger.warning("没有数据可保存")
            return

        if filename is None:
            date_str = datetime.now().strftime('%Y%m%d')
            filename = f'同花顺大涨股_{date_str}.xlsx'

        try:
            df = pd.DataFrame(stocks_data)

            # 重新排列列顺序
            columns_order = [
                '日期', '股票代码', '股票名称', '最新价', '涨跌幅', '涨跌额',
                '成交量(手)', '成交额(万)', '振幅', '最高', '最低',
                '每股收益', '净利润', '营业收入', '涨停原因分析',
                '融资余额(亿)', '融券余额(亿)', '融资融券余额(亿)', '融资融券日期',
                '当日买入量(手)', '当日卖出量(手)', '净买入量(手)',
                '大单净流入(万)', '中单净流入(万)', '小单净流入(万)',
                '股东人数', '人均流通股(股)'
            ]

            # 只保留实际存在的列
            existing_columns = [col for col in columns_order if col in df.columns]
            other_columns = [col for col in df.columns if col not in columns_order]

            df = df[existing_columns + other_columns]

            # 保存到Excel
            df.to_excel(filename, index=False, engine='openpyxl')
            logger.info(f"数据已保存到: {filename}")

            # 同时保存一份CSV格式
            csv_filename = filename.replace('.xlsx', '.csv')
            df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            logger.info(f"数据已保存到: {csv_filename}")

        except Exception as e:
            logger.error(f"保存数据失败: {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("同花顺大涨股数据抓取工具")
    print("=" * 60)

    scraper = ThsStockScraper()

    try:
        # 获取日期
        date_input = input("请输入抓取日期(格式: YYYYMMDD, 回车使用今天): ").strip()
        date = date_input if date_input else None

        logger.info("开始抓取大涨股数据...")

        # 获取大涨股列表
        stocks = scraper.get_rising_stocks(date)

        if not stocks:
            logger.error("未能获取到任何股票数据")
            return

        logger.info(f"成功获取 {len(stocks)} 只大涨股数据")

        # 显示前5只股票
        print("\n前5只大涨股信息:")
        print("-" * 60)
        for i, stock in enumerate(stocks[:5], 1):
            print(f"{i}. {stock['股票名称']}({stock['股票代码']}): {stock['涨跌幅']}")
            if '涨停原因分析' in stock:
                print(f"   涨停原因: {stock['涨停原因分析'][:50]}...")

        # 保存数据
        save = input("\n是否保存数据到文件? (y/n): ").strip().lower()
        if save == 'y':
            scraper.save_to_excel(stocks)

        print("\n数据抓取完成!")

    except KeyboardInterrupt:
        print("\n\n用户中断操作")
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
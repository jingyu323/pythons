# stock_monitor.py
import requests
import pandas as pd
import mysql.connector
from mysql.connector import Error
from bs4 import BeautifulSoup
import json
import time
import logging
import schedule
from datetime import datetime, timedelta
import re
from typing import Dict, List, Optional, Tuple
import random
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
import sys
import os
from dataclasses import dataclass
from enum import Enum

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_monitor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'ruiyi',
    'charset': 'utf8mb4'
}


class StockStatus(Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'


@dataclass
class StockInfo:
    """股票信息数据类"""
    stock_code: str
    stock_name: str
    first_limit_date: str
    industry: str = ''
    limit_reason: str = ''
    concept: str = ''


@dataclass
class DailyData:
    """每日行情数据类"""
    stock_code: str
    trade_date: str
    open_price: float
    close_price: float
    high_price: float
    low_price: float
    volume: int
    amount: float
    change_percent: float
    limit_up_days: int = 0
    is_limit_up: bool = False
    limit_reason: str = ''
    market_cap: float = 0.0
    turnover_rate: float = 0.0
    pe_ratio: float = 0.0
    pb_ratio: float = 0.0


class DBManager:
    """数据库管理器"""

    def __init__(self):
        self.connection = None
        self.connect()

    def connect(self):
        """连接数据库"""
        try:
            self.connection = mysql.connector.connect(**DB_CONFIG)
            logger.info("数据库连接成功")
        except Error as e:
            logger.error(f"数据库连接失败: {e}")
            raise

    def reconnect(self):
        """重连数据库"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
        self.connect()

    def execute_query(self, query: str, params: tuple = None, fetch: bool = False):
        """执行SQL查询"""
        cursor = None
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, params or ())

            if fetch:
                result = cursor.fetchall()
                return result
            else:
                self.connection.commit()
                return cursor.rowcount
        except Error as e:
            logger.error(f"SQL执行失败: {e}, SQL: {query}")
            self.connection.rollback()
            raise
        finally:
            if cursor:
                cursor.close()

    def close(self):
        """关闭数据库连接"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("数据库连接已关闭")


class CaiLianSheScraper:
    """财联社首板涨停股票爬虫"""

    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Referer': 'https://www.cls.cn/',
        }
        self.session.headers.update(self.headers)

    def get_first_limit_up_stocks(self, date: str = None) -> List[StockInfo]:
        """
        获取财联社首板涨停股票
        Args:
            date: 日期，格式为YYYY-MM-DD，默认为今天
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        stocks = []
        try:
            # 财联社涨停板数据页面
            url = f"https://www.cls.cn/search?keyword=首板涨停&type=article&page=1"

            response = self.session.get(url, timeout=30)
            response.encoding = 'utf-8'

            if response.status_code != 200:
                logger.error(f"请求失败，状态码: {response.status_code}")
                return stocks

            # 解析JSON数据
            data = response.json()

            # 这里需要根据财联社实际返回的数据结构进行解析
            # 以下为示例代码，实际需要调整

            if 'data' in data and 'list' in data['data']:
                articles = data['data']['list']

                for article in articles:
                    if '首板' in article.get('title', '') or '首板' in article.get('content', ''):
                        stock_info = self.parse_article_for_stocks(article, date)
                        if stock_info:
                            stocks.extend(stock_info)

            logger.info(f"从财联社获取到 {len(stocks)} 只首板涨停股票")
            return stocks

        except Exception as e:
            logger.error(f"获取首板涨停股票失败: {e}")
            return []

    def parse_article_for_stocks(self, article: Dict, date: str) -> List[StockInfo]:
        """从文章中解析股票信息"""
        stocks = []

        try:
            content = article.get('content', '')
            title = article.get('title', '')

            # 使用正则表达式提取股票信息
            # 格式如：002415.SZ 海康威视
            pattern = r'(\d{6})\.(SZ|SH)\s+([^\s，。]+)'
            matches = re.findall(pattern, content)

            for match in matches:
                stock_code = match[0] + '.' + match[1]
                stock_name = match[2]

                # 提取涨停原因
                limit_reason = self.extract_limit_reason(content, stock_name)

                # 提取所属概念
                concept = self.extract_concept(content)

                stock_info = StockInfo(
                    stock_code=stock_code,
                    stock_name=stock_name,
                    first_limit_date=date,
                    industry=self.get_industry(stock_code),
                    limit_reason=limit_reason,
                    concept=concept
                )
                stocks.append(stock_info)

                logger.debug(f"解析到股票: {stock_name}({stock_code})")

        except Exception as e:
            logger.error(f"解析文章失败: {e}")

        return stocks

    def extract_limit_reason(self, content: str, stock_name: str) -> str:
        """提取涨停原因"""
        reasons = []

        # 常见涨停原因关键词
        reason_keywords = [
            '业绩预增', '政策利好', '概念炒作', '重组', '定增', '订单',
            '产品涨价', '技术突破', '重大合同', '股权转让', '资产注入',
            '行业景气', '资金追捧', '游资炒作', '机构买入', '北向资金',
            '涨停板敢死队', '技术面突破', '消息面刺激'
        ]

        for keyword in reason_keywords:
            if keyword in content:
                reasons.append(keyword)

        # 提取股票名称附近的句子作为原因
        if stock_name in content:
            idx = content.find(stock_name)
            start = max(0, idx - 100)
            end = min(len(content), idx + 100)
            context = content[start:end]

            # 清理上下文
            context = re.sub(r'\s+', ' ', context)
            if len(context) > 50:
                reasons.append(context[:200])

        return '；'.join(reasons) if reasons else '未知原因'

    def extract_concept(self, content: str) -> str:
        """提取所属概念"""
        concepts = []

        # 常见概念关键词
        concept_keywords = [
            '5G', '人工智能', '芯片', '半导体', '新能源汽车', '锂电池',
            '光伏', '风电', '储能', '元宇宙', '数字货币', '区块链',
            '医药', '医疗器械', 'CXO', '消费电子', '白酒', '食品饮料',
            '军工', '航空航天', '券商', '银行', '保险', '房地产',
            '基建', '水泥', '钢铁', '煤炭', '化工', '有色金属'
        ]

        for concept in concept_keywords:
            if concept in content:
                concepts.append(concept)

        return '、'.join(concepts)

    def get_industry(self, stock_code: str) -> str:
        """获取股票所属行业"""
        # 这里可以调用其他接口获取行业信息
        # 暂时返回空字符串，后续可以集成东方财富等行业分类
        return ""


class StockDataFetcher:
    """股票数据获取器"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        })

    def get_daily_data(self, stock_code: str, trade_date: str) -> Optional[DailyData]:
        """
        获取股票日线数据
        使用腾讯财经接口
        """
        try:
            # 腾讯财经接口
            market = 'sz' if stock_code.endswith('.SZ') else 'sh'
            code = stock_code[:6]
            url = f"http://qt.gtimg.cn/q={market}{code}"

            response = self.session.get(url, timeout=10)
            response.encoding = 'gb2312'

            if response.status_code != 200:
                return None

            content = response.text
            # 解析返回的数据格式
            # v_sz000001="1~平安银行~000001~12.34~12.35~12.36..."
            data_pattern = rf'v_{market}{code}="([^"]+)"'
            match = re.search(data_pattern, content)

            if not match:
                return None

            data_str = match.group(1)
            fields = data_str.split('~')

            if len(fields) < 40:
                return None

            # 解析字段
            daily_data = DailyData(
                stock_code=stock_code,
                trade_date=trade_date,
                open_price=float(fields[5]) if fields[5] else 0.0,
                close_price=float(fields[3]) if fields[3] else 0.0,
                high_price=float(fields[33]) if fields[33] else 0.0,
                low_price=float(fields[34]) if fields[34] else 0.0,
                volume=int(fields[36]) if fields[36] else 0,
                amount=float(fields[37]) if fields[37] else 0.0,
                change_percent=float(fields[32]) if fields[32] else 0.0,
                is_limit_up=self.check_limit_up(float(fields[32])),
                market_cap=float(fields[45]) if fields[45] else 0.0,
                turnover_rate=float(fields[38]) if fields[38] else 0.0,
                pe_ratio=float(fields[39]) if fields[39] else 0.0,
                pb_ratio=float(fields[46]) if fields[46] else 0.0
            )

            # 获取涨停原因
            daily_data.limit_reason = self.get_limit_reason_for_date(stock_code, trade_date)

            return daily_data

        except Exception as e:
            logger.error(f"获取股票{stock_code}日线数据失败: {e}")
            return None

    def check_limit_up(self, change_percent: float) -> bool:
        """检查是否涨停"""
        return change_percent >= 9.9

    def get_limit_reason_for_date(self, stock_code: str, date: str) -> str:
        """获取指定日期的涨停原因"""
        # 这里可以调用其他接口获取详细涨停原因
        # 暂时返回空字符串
        return ""

    def get_historical_data(self, stock_code: str, start_date: str, end_date: str) -> List[DailyData]:
        """
        获取历史数据
        使用AKShare库（需要安装）
        """
        try:
            # 如果安装了AKShare
            import akshare as ak

            market = 'sz' if stock_code.endswith('.SZ') else 'sh'
            code = stock_code[:6]

            # 获取历史数据
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )

            daily_data_list = []
            for _, row in df.iterrows():
                daily_data = DailyData(
                    stock_code=stock_code,
                    trade_date=row['日期'].strftime('%Y-%m-%d'),
                    open_price=float(row['开盘']),
                    close_price=float(row['收盘']),
                    high_price=float(row['最高']),
                    low_price=float(row['最低']),
                    volume=int(row['成交量']),
                    amount=float(row['成交额']),
                    change_percent=float(row['涨跌幅']),
                    is_limit_up=self.check_limit_up(float(row['涨跌幅'])),
                    turnover_rate=float(row['换手率']) if '换手率' in row else 0.0
                )
                daily_data_list.append(daily_data)

            # 计算连板天数
            daily_data_list = self.calculate_limit_up_days(daily_data_list)

            return daily_data_list

        except ImportError:
            logger.warning("未安装AKShare，无法获取历史数据")
            return []
        except Exception as e:
            logger.error(f"获取历史数据失败: {e}")
            return []

    def calculate_limit_up_days(self, daily_data_list: List[DailyData]) -> List[DailyData]:
        """计算连板天数"""
        if not daily_data_list:
            return []

        # 按日期排序
        daily_data_list.sort(key=lambda x: x.trade_date)

        consecutive_days = 0
        for i, data in enumerate(daily_data_list):
            if data.is_limit_up:
                consecutive_days += 1
            else:
                consecutive_days = 0
            data.limit_up_days = consecutive_days

        return daily_data_list


class StockMonitor:
    """股票监控主类"""

    def __init__(self):
        self.db = DBManager()
        self.scraper = CaiLianSheScraper()
        self.fetcher = StockDataFetcher()
        self.executor = ThreadPoolExecutor(max_workers=5)

    def save_first_limit_stocks(self, stocks: List[StockInfo]):
        """保存首板涨停股票到数据库"""
        if not stocks:
            return

        for stock in stocks:
            try:
                query = """
                        INSERT INTO first_limit_up_stocks
                        (stock_code, stock_name, first_limit_date, industry, limit_reason, concept)
                        VALUES (%s, %s, %s, %s, %s, %s) ON DUPLICATE KEY \
                        UPDATE \
                            update_time = CURRENT_TIMESTAMP, \
                            limit_reason = \
                        VALUES (limit_reason), concept = \
                        VALUES (concept) \
                        """

                params = (
                    stock.stock_code,
                    stock.stock_name,
                    stock.first_limit_date,
                    stock.industry,
                    stock.limit_reason,
                    stock.concept
                )

                self.db.execute_query(query, params)
                logger.info(f"保存股票: {stock.stock_name}({stock.stock_code})")

                # 创建监控任务
                self.create_monitor_task(stock)

            except Exception as e:
                logger.error(f"保存股票{stock.stock_code}失败: {e}")

    def create_monitor_task(self, stock: StockInfo):
        """创建监控任务"""
        try:
            end_date = (datetime.strptime(stock.first_limit_date, '%Y-%m-%d') +
                        timedelta(days=180)).strftime('%Y-%m-%d')

            query = """
                    INSERT INTO monitor_tasks
                        (stock_code, start_date, end_date, status)
                    VALUES (%s, %s, %s, 'pending') ON DUPLICATE KEY \
                    UPDATE \
                        update_time = CURRENT_TIMESTAMP \
                    """

            params = (stock.stock_code, stock.first_limit_date, end_date)
            self.db.execute_query(query, params)

            logger.info(f"创建监控任务: {stock.stock_code} ({stock.first_limit_date} 至 {end_date})")

        except Exception as e:
            logger.error(f"创建监控任务失败: {e}")

    def save_daily_data(self, daily_data: DailyData):
        """保存每日行情数据"""
        try:
            query = """
                    INSERT INTO stock_daily_data
                    (stock_code, trade_date, open_price, close_price, high_price, low_price,
                     volume, amount, change_percent, limit_up_days, is_limit_up, limit_reason,
                     market_cap, turnover_rate, pe_ratio, pb_ratio)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY \
                    UPDATE \
                        open_price = \
                    VALUES (open_price), close_price = \
                    VALUES (close_price), high_price = \
                    VALUES (high_price), low_price = \
                    VALUES (low_price), volume = \
                    VALUES (volume), amount = \
                    VALUES (amount), change_percent = \
                    VALUES (change_percent), limit_up_days = \
                    VALUES (limit_up_days), is_limit_up = \
                    VALUES (is_limit_up), limit_reason = \
                    VALUES (limit_reason), market_cap = \
                    VALUES (market_cap), turnover_rate = \
                    VALUES (turnover_rate), pe_ratio = \
                    VALUES (pe_ratio), pb_ratio = \
                    VALUES (pb_ratio), update_time = CURRENT_TIMESTAMP \
                    """

            params = (
                daily_data.stock_code,
                daily_data.trade_date,
                daily_data.open_price,
                daily_data.close_price,
                daily_data.high_price,
                daily_data.low_price,
                daily_data.volume,
                daily_data.amount,
                daily_data.change_percent,
                daily_data.limit_up_days,
                1 if daily_data.is_limit_up else 0,
                daily_data.limit_reason,
                daily_data.market_cap,
                daily_data.turnover_rate,
                daily_data.pe_ratio,
                daily_data.pb_ratio
            )

            self.db.execute_query(query, params)

        except Exception as e:
            logger.error(f"保存每日数据失败: {e}")

    def update_task_progress(self, stock_code: str, progress: int, status: StockStatus):
        """更新任务进度"""
        try:
            query = """
                    UPDATE monitor_tasks
                    SET progress    = %s, \
                        status      = %s, \
                        update_time = CURRENT_TIMESTAMP
                    WHERE stock_code = %s \
                    """

            params = (progress, status.value, stock_code)
            self.db.execute_query(query, params)

        except Exception as e:
            logger.error(f"更新任务进度失败: {e}")

    def monitor_stock(self, stock_code: str, start_date: str, end_date: str):
        """监控单只股票"""
        try:
            logger.info(f"开始监控股票: {stock_code}")
            self.update_task_progress(stock_code, 0, StockStatus.RUNNING)

            # 获取历史数据
            historical_data = self.fetcher.get_historical_data(
                stock_code, start_date, end_date
            )

            if historical_data:
                # 保存数据
                for i, daily_data in enumerate(historical_data):
                    self.save_daily_data(daily_data)

                    # 更新进度
                    progress = int((i + 1) / len(historical_data) * 100)
                    self.update_task_progress(stock_code, progress, StockStatus.RUNNING)

                    time.sleep(0.1)  # 避免请求过快

                logger.info(f"完成监控股票: {stock_code}, 共{len(historical_data)}天数据")
                self.update_task_progress(stock_code, 100, StockStatus.COMPLETED)
            else:
                logger.warning(f"未获取到股票{stock_code}的历史数据")
                self.update_task_progress(stock_code, 0, StockStatus.FAILED)

        except Exception as e:
            logger.error(f"监控股票{stock_code}失败: {e}")
            self.update_task_progress(stock_code, 0, StockStatus.FAILED)

    def run_daily_collection(self, date: str = None):
        """运行每日收集任务"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        logger.info(f"开始每日数据收集，日期: {date}")

        try:
            # 获取首板涨停股票
            stocks = self.scraper.get_first_limit_up_stocks(date)

            # 保存股票信息
            self.save_first_limit_stocks(stocks)

            logger.info(f"每日收集完成，获取到{len(stocks)}只股票")

        except Exception as e:
            logger.error(f"每日收集失败: {e}")

    def run_monitor_tasks(self):
        """运行监控任务"""
        try:
            # 获取待处理的任务
            query = """
                    SELECT stock_code, start_date, end_date
                    FROM monitor_tasks
                    WHERE status IN ('pending', 'running')
                      AND end_date >= CURDATE() \
                    """

            tasks = self.db.execute_query(query, fetch=True)

            if not tasks:
                logger.info("没有待处理的监控任务")
                return

            logger.info(f"开始处理{len(tasks)}个监控任务")

            # 使用线程池并发处理
            futures = []
            for task in tasks:
                future = self.executor.submit(
                    self.monitor_stock,
                    task['stock_code'],
                    task['start_date'].strftime('%Y-%m-%d'),
                    task['end_date'].strftime('%Y-%m-%d')
                )
                futures.append(future)

            # 等待所有任务完成
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"任务执行失败: {e}")

            logger.info("所有监控任务处理完成")

        except Exception as e:
            logger.error(f"运行监控任务失败: {e}")

    def export_data_to_csv(self, stock_code: str = None):
        """导出数据到CSV"""
        try:
            if stock_code:
                query = """
                        SELECT * \
                        FROM stock_daily_data
                        WHERE stock_code = %s
                        ORDER BY trade_date \
                        """
                params = (stock_code,)
                filename = f"stock_data_{stock_code}.csv"
            else:
                query = "SELECT * FROM stock_daily_data ORDER BY stock_code, trade_date"
                params = None
                filename = "stock_data_all.csv"

            data = self.db.execute_query(query, params, fetch=True)

            if data:
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                logger.info(f"数据已导出到: {filename}")
            else:
                logger.warning("没有数据可导出")

        except Exception as e:
            logger.error(f"导出数据失败: {e}")

    def generate_report(self):
        """生成统计报告"""
        try:
            # 获取统计信息
            query = """
                    SELECT COUNT(DISTINCT stock_code)                       as total_stocks, \
                           COUNT(*)                                         as total_records, \
                           MIN(trade_date)                                  as start_date, \
                           MAX(trade_date)                                  as end_date, \
                           AVG(change_percent)                              as avg_change, \
                           SUM(CASE WHEN is_limit_up = 1 THEN 1 ELSE 0 END) as total_limit_ups
                    FROM stock_daily_data \
                    """

            stats = self.db.execute_query(query, fetch=True)[0]

            report = f"""
            ===== 股票监控系统报告 =====

            统计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

            监控概况:
            - 监控股票数量: {stats['total_stocks']}
            - 数据记录总数: {stats['total_records']}
            - 监控时间范围: {stats['start_date']} 至 {stats['end_date']}
            - 平均涨跌幅: {stats['avg_change']:.2f}%
            - 涨停次数总计: {stats['total_limit_ups']}

            最近首板股票:
            """

            # 获取最近的首板股票
            recent_query = """
                           SELECT stock_code, stock_name, first_limit_date, limit_reason
                           FROM first_limit_up_stocks
                           ORDER BY first_limit_date DESC LIMIT 10 \
                           """

            recent_stocks = self.db.execute_query(recent_query, fetch=True)

            for stock in recent_stocks:
                report += f"\n- {stock['stock_name']}({stock['stock_code']}) " \
                          f"首板: {stock['first_limit_date']}\n  原因: {stock['limit_reason'][:100]}..."

            # 写入报告文件
            with open('stock_monitor_report.txt', 'w', encoding='utf-8') as f:
                f.write(report)

            print(report)
            logger.info("报告已生成")

        except Exception as e:
            logger.error(f"生成报告失败: {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("财联社首板涨停股票监控系统")
    print("=" * 60)

    monitor = StockMonitor()

    try:
        while True:
            print("\n请选择操作:")
            print("1. 运行每日收集（获取首板涨停股票）")
            print("2. 运行监控任务（跟踪股票半年数据）")
            print("3. 导出数据到CSV")
            print("4. 生成统计报告")
            print("5. 手动添加监控股票")
            print("0. 退出")

            choice = input("\n请输入选项: ").strip()

            if choice == '1':
                date = input("请输入日期(YYYY-MM-DD, 回车使用今天): ").strip()
                date = date if date else None
                monitor.run_daily_collection(date)

            elif choice == '2':
                monitor.run_monitor_tasks()

            elif choice == '3':
                stock_code = input("请输入股票代码(回车导出所有): ").strip()
                stock_code = stock_code if stock_code else None
                monitor.export_data_to_csv(stock_code)

            elif choice == '4':
                monitor.generate_report()

            elif choice == '5':
                stock_code = input("请输入股票代码(如: 000001.SZ): ").strip()
                stock_name = input("请输入股票名称: ").strip()
                first_date = input("请输入首板日期(YYYY-MM-DD): ").strip()

                if stock_code and stock_name and first_date:
                    stock = StockInfo(
                        stock_code=stock_code,
                        stock_name=stock_name,
                        first_limit_date=first_date
                    )
                    monitor.save_first_limit_stocks([stock])
                    print("股票已添加")
                else:
                    print("输入信息不完整")

            elif choice == '0':
                print("感谢使用，再见！")
                break

            else:
                print("无效选项，请重新选择")

    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        monitor.executor.shutdown(wait=True)
        monitor.db.close()


if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # 检查依赖
    try:
        import mysql.connector
        import pandas
    except ImportError as e:
        print(f"缺少依赖包: {e}")
        print("请安装所需依赖: pip install mysql-connector-python pandas requests beautifulsoup4")
        sys.exit(1)

    main()
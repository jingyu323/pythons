# config.py
# 配置参数

# 爬虫设置
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
DELAY_BETWEEN_REQUESTS = 1  # 请求间隔秒数

# 数据筛选条件
MIN_RISE_PERCENT = 9.5  # 最小涨幅百分比
MAX_STOCKS = 50  # 最多获取的股票数量

# 输出设置
OUTPUT_DIR = "./output"
LOG_DIR = "./logs"
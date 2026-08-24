import akshare as ak

import pandas as pd

# 先用 akshare 快速筛选热门股，再用 tushare 查财务

print(pd.__version__)

df = ak.stock_zh_a_daily(symbol="sh600519", adjust="qfq")

print(df.tail())

# 获取沪深300指数

# index_df = ak.index_zh_a_hist(symbol="sh000300")
# 获取融资融券余额
# print(index_df)
margin_df = ak.stock_margin_szse() # 深市两融汇总

print(margin_df)

hot_stocks = ak.stock_zh_a_spot_em() .sort_values(by='涨跌幅', ascending=False).head(10)

print(hot_stocks)

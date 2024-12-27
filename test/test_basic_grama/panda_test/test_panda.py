from pandas import DataFrame,Series
import pandas as pd
import  tushare as ts
import  numpy as np

print(ts.__version__)
print(pd.__version__)

# df = ts.get_k_data(code="600519",start='2008-01-01')
# print(df)


s=Series(data=[1,2,3,'ssss'],index=['a','b','c','d'])

print(s)

df = DataFrame(data=[[1,2,3],[4,5,6]])

print(df)

df2 = DataFrame(data=np.random.randint(60,100,size=(8,4)),columns=['a','b','c','d'])

print(df2)

# 使用索引机制取值
print(df2['a'])


df3 = ts.get_k_data(code='600519',start='2007-01-01')

print(df3)
print("====------")
print()
print(df3.loc[(df3['open']-df3['close'])/df3['open'] > 0.03] )
print("====")
print(df3.loc[(df3['open']-df3['close'])/df3['open'] > 0.03]['date'])



"""
1、concat
用concat是一种基本的合并方式。而且concat中有很多参数可以调整，合并成你想要的数据形式。axis来指明合并方向。axis=0是预设值，因此未设定任何参数时，函数默认axis=0。（0表示上下合并，1表示左右合并）

"""
import numpy as np
import pandas as pd

# 定义资料集
df1 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'])
df2 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['a', 'b', 'c', 'd'])
df3 = pd.DataFrame(np.ones((3, 4)) * 2, columns=['a', 'b', 'c', 'd'])

print(df1)
print(df2)
print(df3)

# concat纵向合并 重置后的index为0，1，……8
res = pd.concat([df1, df2, df3], axis=0, ignore_index=True)

# 打印结果
print(res)

# 水平合并
res = pd.concat([df1, df2, df3], axis=1)
print(res)
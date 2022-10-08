import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib.font_manager as font_manager
import pandas as pd
#显示matplotlib生成的图形

df = pd.read_json('work/stars_info.json')
heights=df['height']
arrs = heights.values

arrs = [x for x in arrs if not pd.isnull(x)]
for i in range(len(arrs)):
    arrs[i] = float(arrs[i])

#pandas.cut用来把一组数据分割成离散的区间。比如有一组年龄数据，可以使用pandas.cut将年龄数据分割成不同的年龄段并打上标签。bins是被切割后的区间.
bin=[0,165,170,180]
se1=pd.cut(arrs,bin)

#pandas的value_counts()函数可以对Series里面的每个值进行计数并且排序。
pd.value_counts(se1)

labels =  '165~170cm','<=165cm', '>170cm'
sizes = pd.value_counts(se1)
print(sizes)

explode = (0.1, 0.1, 0,)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.savefig('pie_result03.jpg')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib.font_manager as font_manager
with open('work/stars_info.json', 'r', encoding='UTF-8') as file:
    json_array = json.loads(file.read())

# 绘制选手体重分布饼状图
weights = []
counts = []

for star in json_array:
    if 'weight' in dict(star).keys():
        weight = float(star['weight'][0:2])
        weights.append(weight)
print(weights)

size_list = []
count_list = []

size1 = 0
size2 = 0
size3 = 0
size4 = 0

for weight in weights:
    if weight <= 45:
        size1 += 1
    elif 45 < weight <= 50:
        size2 += 1
    elif 50 < weight <= 55:
        size3 += 1
    else:
        size4 += 1

labels = '<=45kg', '45~50kg', '50~55kg', '>55kg'

sizes = [size1, size2, size3, size4]
explode = (0.2, 0.1, 0, 0)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True)
ax1.axis('equal')
plt.savefig('pie_result01.jpg')
plt.show()


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
    arrs[i] = float(arrs[i][0:3])

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
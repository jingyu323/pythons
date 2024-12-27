
import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib.font_manager as font_manager
#显示matplotlib生成的图形


with open('work/stars_info.json', 'r', encoding='UTF-8') as file:
         json_array = json.loads(file.read())

#绘制选手年龄分布柱状图,x轴为年龄，y轴为该年龄的小姐姐数量
birth_days = []
for star in json_array:
    if 'birth_day' in dict(star).keys():
        birth_day = star['birth_day']
        if len(birth_day) == 4:
            birth_days.append(birth_day)

birth_days.sort()
print(birth_days)

birth_days_list = []
count_list = []

for birth_day in birth_days:
    if birth_day not in birth_days_list:
        count = birth_days.count(birth_day)
        birth_days_list.append(birth_day)
        count_list.append(count)

print(birth_days_list)
print(count_list)

# 设置显示中文
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
plt.figure(figsize=(15,8))
plt.bar(range(len(count_list)), count_list,color='r',tick_label=birth_days_list,
            facecolor='#9999ff',edgecolor='white')

# 这里是调节横坐标的倾斜度，rotation是度数，以及设置刻度字体大小
plt.xticks(rotation=45,fontsize=20)
plt.yticks(fontsize=20)

plt.legend()
plt.title('''《乘风破浪的姐姐》参赛嘉宾''',fontsize = 24)
plt.savefig('bar_result01.jpg')
plt.show()

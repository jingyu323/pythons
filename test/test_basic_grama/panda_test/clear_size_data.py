import os

import pandas

path='.'


def get_directory_size(directory):
    """
    获取目录的大小（以MB为单位）。
    """
    size = os.path.getsize(directory)
    return size / 1024 / 1024  # 转换为MB


for file in os.listdir(path):

    print(get_directory_size(file))
    if os.path.isdir(file):

        print(file.__sizeof__())
        os.path.getsize(file)

dataFrame = pandas.DataFrame(
    data=[
        [60, 78, 92, 85],
        [70, 68, 95, 76],
        [88, 98, 83, 87]
    ],
    index=['小明', '小红', '小强', ],
    columns=['语文', '数学', '英语', '化学'],
    dtype=float,
    copy=True
)
print(dataFrame)



dataFrame = pandas.DataFrame(
    data = {
        '语文': [60, 78, 92, 85],
        '数学': [70, 68, 95, 76],
        '英语': [88, 98, 83, 87],
    },
    index = ['小明', '小红', '小强', '小美'],
    dtype = float,
    copy = True
)
print(dataFrame)

dataFrame = pandas.DataFrame(
    data=[
        [60, 78, 92, 85],
        [70, 68, 95, 76],
        [88, 98, 83, 87]
    ],
    index=['小明', '小红', '小强', ],
    columns=['语文', '数学', '英语', '化学'],
    dtype=float,
    copy=False
)
dataFrame2 = pandas.DataFrame(dataFrame, copy=False)
print(dataFrame2)
dataFrame2['语文'] = [0, 0, 0]
print(dataFrame2)
print(dataFrame)

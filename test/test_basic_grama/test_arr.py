import bisect

list=[1,2,3,4]

tem = list


tem[0]=3

print(list)


dp = [1] * len(list)

print(dp)

# index = bisect.bisect_left(res, arr[i])

import numpy as np
mylist = [1,2,3]
print(tuple(mylist))
iarray = np.array(tuple(mylist))
print( iarray)

names = ['a', 'b', 'c', 'd', 'b']
names.remove('b')

import numpy as np

# 将列表转换为numpy的数组
a = np.array(["a", "b", "c", "a", "d", "a"])
# 获取元素的下标位置
eq_letter = np.where(a == "a")
print(eq_letter[0])  # [0 3 5]
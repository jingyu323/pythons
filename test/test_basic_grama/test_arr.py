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


nums = [1,5,7,8,9,6,3,11,20,17]
n, sum = len(nums), sum(nums)
isOK = [[False]*(sum//2+1) for _ in range(n//2+1)]
isOK[0][0]=True
for k in range(1,n+1):
    for i in range(min(k,n//2),0,-1):
        for v in range(1,sum//2+1):
            if v >= nums[k-1] and isOK[i-1][v-nums[k-1]]:
                isOK[i][v] = True
for i in range(n//2+1):
    for j in range(sum//2+1):
        print(isOK[i][j],end=' ')

from collections import Counter

a = [29,36,57,12,79,43,23,56,28,11,14,15,16,37,24,35,17,24,33,15,39,46,52,13]
b = dict(Counter(a))

print ([key for key,value in b.items()if value > 1]) #只展示重复元素
print ({key:value for key,value in b.items()if value > 1}) #展现重复元素和重复次数
V =10
n=16
# [0 for j in range(V + 1)]  一维数组的内容
# for i in range(n + 1) 元素的  二维数组的个使用
dp = [[0 for j in range(V + 1)] for i in range(n + 1)]

print(dp)
a = {'device_type': 'cisco_ios', 'username': 'admin', 'password': 'cisco'}



b = {'name': 'r1'}

b.update(a)

print(b)

dp = [[0 for i in range(V+1)] for j in range(n+1)]



N =int(input())
input_res=set()
for i in  range(N):
    input_res.add(int(input()))

lee =sorted(input_res)
for i in lee:
    print(i)



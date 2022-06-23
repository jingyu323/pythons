
from collections import Counter

a = [29,36,57,12,79,43,23,56,28,11,14,15,16,37,24,35,17,24,33,15,39,46,52,13]
b = dict(Counter(a))

print ([key for key,value in b.items()if value > 1]) #只展示重复元素
print ({key:value for key,value in b.items()if value > 1}) #展现重复元素和重复次数

N =int(input())
input_res=set()
for i in  range(N):
    input_res.add(int(input()))

lee =sorted(input_res)
for i in lee:
    print(i)




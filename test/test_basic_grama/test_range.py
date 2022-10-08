"""
python 遍历数组主要有两种：
1.借助于 range

people = ["李白","杜甫","我"]
for i in range(0, len(people)):
    print(people[i])

2. for in：

for i in range(10):
    print(i)


3.倒序： 借助于for in range

for j in range(结束, 0, -1):

"""
for i in range(10):
    print(i)


for i in range(1, 20, 2):
    print(i, end=" ")

for i in range(1, 6):
    for j in range(6-i, 0, -1):
        print("*", end=" ")
    print ()

d = {'a':'apple', 'b':'banana', 'c':'car', 'd': 'desk'}
for key in d:
    # 遍历字典时遍历的是键
    print(key, d.get(key))
# for key, value in d.items():
# 上下两种方式等价 d.items() <=> dict.items(d)
for key, value in dict.items(d):
    print(key, value)


try:
    print(2)
except Exception:
    print()

linesTheta = [ 2.4958208, 2.5307274, 2.5132742, 2.5307274, 2.5830872, 2.5481806]
list_same = []
for i in linesTheta:
    address_index = [x for x in range(len(linesTheta)) if linesTheta[x] == i]
    list_same.append([i, address_index])
dict_address = dict(list_same)

index = []
for values in dict_address.values():
    if len(values) == 2:
        index = values

print(index[1])
# print(dict_address)
print(dict_address)

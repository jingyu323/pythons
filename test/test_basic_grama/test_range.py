
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
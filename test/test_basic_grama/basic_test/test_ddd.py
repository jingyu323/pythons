from functools import cmp_to_key

sss="sjjsjsjjs"
sarr=list(sss)
for ss in sarr:
    print(ss)

count = 30888 #麦叔粉丝数

# 关注
def guanzhu():
    global count
    count = count + 1
    print('麦叔的粉丝数是{}'.format(count))

# 取关
def quguan():
    global count
    count = count - 1
    print('麦叔的粉丝数是{}'.format(count))

guanzhu()

quguan()


countsss = "ssssss" #麦叔粉丝数

# 关注
def guanzhu3333(cou):
    global countsss
    countsss = countsss + "------"
    print('----{}'.format(countsss))


guanzhu3333(countsss)

print("countsss："+countsss)
def numeric_compare(x, y):
    return x - y
list =[5, 2, 4, 1, 3]
# python3
print(sorted([5, 2, 4, 1, 3], key=cmp_to_key(numeric_compare)))
print("---------------")
list.sort()

print(list)
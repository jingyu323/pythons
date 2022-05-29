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
def guanzhu3333():
    global countsss
    countsss = countsss + "------"
    print('----{}'.format(countsss))


guanzhu3333()

print("countsss："+countsss)


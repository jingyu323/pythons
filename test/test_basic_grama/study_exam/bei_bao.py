from itertools import *
u = []
v = []

s = int(input("背包容量："))
t = int(input("物品数量："))
for i in range(t):
    u.append(int(input(f"第{i+1}个物品重量：")))
    v.append(int(input(f"第{i+1}个物品价值：")))
a = []
for i in range(len(u)):
    q = [u[i], v[i]]
    a.append(q)
a=list(permutations(a))

al=[]
for i in a:
    al.append(list(i))

print(al)

gdl = []
for x in al:  # x是一种情况，示例：[[1(重量),2(价值)],[1(重量),2(价值)]]
    jz = 0
    wt = 0
    jq = len(x) - 1
    for y in x:  # y是一组数据，示例：[1(重量),2(价值)]

        jz += y[1]
        wt += y[0]

    while True:

        if wt > s:
            wt -= x[jq][0]
            jz -= x[jq][1]
            jq -= 1

        if wt <= s:
            gdl.append(jz)
            break

print(max(gdl))
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
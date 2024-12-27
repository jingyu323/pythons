start=[]
while True:
    try:
        a = input()
        if a == ""  :
            break
        start.append(a)
    except:
        break

def jiequ(x):
    x =x.split("\\")
    if len(x[-1]) >20:
        x =(x[-1])[-20:]
    else:
        x =x[-1]
    return x
i =0
b =[]
for i in start:
    i =jiequ(i)
    b.append(i)
j =0
x =0
c =[]
for j in b:
    x = b.count(j)
    j = str(j) + " " + str(x)
    if j not in c:
        c.append(j)


if  len(c) >= 8:
    ind=len(c) -8
    for x1 in range(8):
        print(c[ ind +x1])
else:
    for c1 in c:
        print(c1)
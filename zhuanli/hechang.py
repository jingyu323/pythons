N =int(input())
intpu_str=input()
print(intpu_str)
h =intpu_str.split()
print(h)
def shi(h1):
    i =0
    for i in h1[:-1]:
        if int(i) >= int(i+1):
            h1.remove(i)
    return len(h1)
def mo(h1):
    i =0
    for i in h1[:-1]:
        if int(i) <= int(i+1):
            h1.remove(i)
    return len(h1)
S =max(h)
print(S)
x =h.index(S)
hy =h[:x]
my =h[x+1:]
renshu =N-int(shi(hy))-int(mo(my))
print("sss:"+str(renshu))
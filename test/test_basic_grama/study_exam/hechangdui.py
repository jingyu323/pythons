N = int(input())
h = input().split(" ")
print(h)

int_h = []
int_max_index_arr=[]
for ch in h:
    int_h.append(int(ch))
h = int_h
def shi(h1):
    i = h1[0]
    arr_len = len(h1)
    tem_remove_arr = []
    for index in range( 1,arr_len) :
        if i < h1[index]:
            i = h1[index]
        else:
            tem_remove_arr.append(h1[index])
    for el in tem_remove_arr:
        h1.remove(el)

    print(h1)
    return   len(h1)


def mo(h1):
    i = h1[0]
    arr_len = len(h1)
    tem_remove_arr = []
    for index in range(1,arr_len):
         if  i > h1[index] :
             i = h1[index]
         else:
             tem_remove_arr.append(h1[index])
    for el in tem_remove_arr:
        h1.remove(el)
    print(h1)
    return  len(h1)


S = max(h)
print(S)
tm_h=[ x for x in h]
while True:
    try:
        x = tm_h.index(S)
    except Exception:
        break
    if x > 0:
        int_max_index_arr.append(x)
        tm_h = tm_h[x+1:]
    else:
        break
renshu=10001
for index in  int_max_index_arr:


    hy = h[:index]
    print(hy)
    my = h[index + 1:]
    print(my)
    left = N - int(shi(hy)) - int(mo(my)) - 1
    renshu =  min(renshu,left)
    print(renshu)
print(renshu)



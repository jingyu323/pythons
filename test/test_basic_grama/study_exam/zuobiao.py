num = input().split(";")
x=0
y=0
for st in num:
    if st == "" or  not st[1:].isdigit() :
        continue

    if st[0] == "A":
        x =x- int(st[1:])
    elif st[0] == "D":
        x =x+ int(st[1:])
    elif st[0] == "W":
        y = y + int(st[1:])
    elif st[0] == "S":
        y = y - int(st[1:])
print(str(x) + "," + str(y))
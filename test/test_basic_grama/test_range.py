
for i in range(10):
    print(i)


for i in range(1, 20, 2):
    print(i, end=" ")

for i in range(1, 6):
    for j in range(6-i, 0, -1):
        print("*", end=" ")
    print ()
n=int(input())


res = []
rowindex=0
for index in  range(n):
    row = []
    res.append(row)

    for num in range(n - index):  # 这里主要是为输出做的格式处理
        if num == 0 and index != 0:
            em=""
            for i  in  range(rowindex+1):
                em= em+ " "
            row.insert(0,em)
        else:
            row.insert(0," ")
    row.append(1)
    for i in range(1,index):
        el = res[index-1][n - index+i]+res[index-1][n - index+i+1]
        row.append(el)

    if index >0:
        row.append(1)

    for num in row:
        print(num, end=" ")
    print()
    rowindex = rowindex +1




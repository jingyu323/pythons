"""

例：
某工厂预计明年有A、B、C、D四个新建项目，每个项目的投资额Wk及其投资后的收益Vk如下表所示，投资总额为30万元，如何选择项目才能使总收益最大？
"""



w = [0, 15, 10, 12, 8];
v = [0, 12, 8, 9, 5]



c = 30;
n = 4;
v_len = len(v)

dp = [[0 for j in range(c+1)] for i in range(n+1)]
item = [0 for j in range (5)]
def findWaht(i, j):
    for i in range(i,0,-1):
        if dp[i][j] == dp[i-1][j]:
            item[i] = 0
        else:
            item[i] = 1
            j = j-w[i]
    # if dp[1][j] > 0:
    #     item[1] = 1
    # else:
    #     item[1] = 0

    print(item)







def findMin():
    global  dp
    for row in range(1, n + 1):
        for col in range(c + 1):
            print(row ,col)
            if col >= w[row]:
                res  = max(dp[row-1][col],dp[row -1][col -w[row] ]+ v[row])
                dp[row][col] = res
                print("res =",res)
                print(" dp[row][col] =", dp[row][col])
            else:
                print("sss else:", dp[row][col])
                dp[row][col]= dp[row-1][col]

print(dp)




findMin()

print(dp)
findWaht(4, 30)





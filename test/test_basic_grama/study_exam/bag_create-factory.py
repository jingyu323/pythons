"""

例：
某工厂预计明年有A、B、C、D四个新建项目，每个项目的投资额Wk及其投资后的收益Vk如下表所示，投资总额为30万元，如何选择项目才能使总收益最大？
"""
dp = []
item = []
w = []
v = []


def findWaht(i, j):
    print("ss")

    if i >= 0:
        if dp[i][j] == dp[i - 1][j]:
            item[i] = 0;
            findWaht(i - 1, j);
    else:
        if j - w[i] >= 0 and dp[i][j] == dp[i - 1][j - w[i]] + v[i]:
            item[i] = 1;
            findWaht(i - 1, j - w[i]);

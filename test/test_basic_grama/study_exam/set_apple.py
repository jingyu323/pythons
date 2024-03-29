"""
描述
把m个同样的苹果放在n个同样的盘子里，允许有的盘子空着不放，问共有多少种不同的分法？
注意：如果有7个苹果和3个盘子，（5，1，1）和（1，5，1）被视为是同一种分法。

数据范围：0 \le m \le 10 \0≤m≤10 ，1 \le n \le 10 \1≤n≤10 。

输入描述：
输入两个int整数

输出描述：
输出结果，int型

示例1
输入：
7 3
复制
输出：
8


"""


def putApple(m, n):
    if m == 0 or n == 1:
        return 1
    elif n > m:
        return putApple(m, m)
    else:
        return putApple(m, n - 1) + putApple(m - n, n)


n, m = map(int, input().split())
print(putApple(n, m))
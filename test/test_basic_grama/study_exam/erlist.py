"""
原题链接：NC145 01背包

https://www.nowcoder.com/practice/2820ea076d144b30806e72de5e5d4bbf?tpId=196

示例1
输入：
10,2,[[1,3],[10,4]]
1
返回值：4
已知一个背包最多能容纳体积之和为v的物品

现有 n 个物品，第 i 个物品的体积为 vi , 重量为 wi

求当前背包最多能装多大重量的物品?

数据范围： 1 \le v \le 10001≤v≤1000 ， 1 \le n \le 10001≤n≤1000 ， 1 \le v_i \le 10001≤v
i
​
 ≤1000 ， 1 \le w_i \le 10001≤w
i
​
 ≤1000

进阶 ：O(n \cdot v)O(n⋅v)
示例1
输入：
10,2,[[1,3],[10,4]]
复制
返回值：
4
复制
说明：
第一个物品的体积为1，重量为3，第二个物品的体积为10，重量为4。只取第二个物品可以达到最优方案，取物重量为4
示例2
输入：
10,2,[[1,3],[9,8]]
复制
返回值：
11
复制
说明：
两个物品体积之和等于背包能装的体积，所以两个物品都取是最优方案

# 计算01背包问题的结果
# @param V int整型 背包的体积
# @param n int整型 物品的个数
# @param vw int整型二维数组 第一维度为n,第二维度为2的二维数组,vw[i][0],vw[i][1]分别描述i+1个物品的vi,wi
# @return int整型
"""

class Solution:
    def knapsack(self , V , n , vw ):
        # write code here
        '''dp是一个矩阵，行是考虑的物品数，分别是0，1，2； 列是背包的可用空间分别是0，1，2，。。。，10；
        而对应位置的值就是给定条件下可以放的最大重量'''
        dp = [[0 for j in range(V+1)] for i in range(n+1)]
        '''这里给vw的index0插入了一个【0，0】是为了之后代码遍历方便点，这样index0 表示考虑0件物品'''
        vw.insert(0,[0,0])
        '''对每一行进行遍历，注意，行的index就是代表考虑最近的index件物品'''
        for row in range(1,n+1):
            for col in range(V+1):
                if vw[row][0]> col:
                    dp[row][col] = dp[row-1][col]
                else:
                    dp[row][col] = max(dp[row-1][col], dp[row-1][col-vw[row][0]] + vw[row][1])
        return dp[n][V]
# 最后一块石头的重量 II LeetCode刷题之路：1049. 最后一块石头的重量 II
"""
targetSum =  sum // 2 ; 整除
 for i in range(targetSum,0,-1):  反向 递减

array = [  0 for j in range (targetSum+1)]  初始化数据
  dp = [[0 for j in range(V+1)] for i in range(n+1)]


m[i-1][j-w[i]]+v[i]

if(j>=w[i])
    m[i][j]=max(m[i-1][j],m[i-1][j-w[i]]+v[i]);
else
    m[i][j]=m[i-1][j];
如果拿取，m[ i ][ j ]=m[ i-1 ][ j-w[ i ] ] + v[ i ]。
 这里的m[ i-1 ][ j-w[ i ] ]指的就是考虑了i-1件物品，背包容量为j-w[i]时的最大价值，也是相当于为第i件物品腾出了w[i]的空间。

"""

def lastStoneWeightII(stones):
    sum = 0;
    st_len = len(stones)
    for i in range(st_len) :
        sum += stones[i];

    targetSum =  sum // 2 ;
    array = [  0 for j in range (targetSum+1)]

    for  j in range(st_len):
        for i in range(targetSum,0,-1):
            without_re = array[i]
            if stones[j] <= i:
                with_re =stones[j] + array[i - stones[j]]
            else:
                with_re = array[i]
            array[i] = max(without_re, with_re);


    return sum - 2 * array[targetSum];


stones = [31,26,33,21,40]
res =lastStoneWeightII(stones)
print(res)

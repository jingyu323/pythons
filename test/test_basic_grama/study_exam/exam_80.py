from functools import cmp_to_key

"""
  
  /*
        给定参数n,从1到n会有n个整数:1,2,3,...,n,
        这n个数字共有n!种排列.
      按大小顺序升序列出所有排列的情况,并一一标记,
      当n=3时,所有排列如下:
      "123" "132" "213" "231" "312" "321"
      给定n和k,返回第k个排列.

      输入描述:
        输入两行，第一行为n，第二行为k，
        给定n的范围是[1,9],给定k的范围是[1,n!]。
      输出描述：
        输出排在第k位置的数字。

      实例1：
        输入:
          3
          3
        输出：
          213
        说明
          3的排列有123,132,213...,那么第三位置就是213

      实例2：
        输入
          2
          2
        输出：
          21
        说明
          2的排列有12,21，那么第二位置的为21
"""

def exam():
    k = int(input())
    n = int(input())

    candidates = []
    factorials = [n+1]

    factorials[0] = 1;
    fact = 1;

    for index  in range(1,n):
        candidates.append(index);
        fact *= index;
        factorials[index] = fact;

    k -= 1;
    for index in range(n-1,0,-1):
        index_fax = k / factorials[index];
        candidates.remove(index_fax)
        k -= index * factorials[index]


if __name__ == '__main__':
    exam()

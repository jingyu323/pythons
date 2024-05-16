from functools import cmp_to_key

"""
    /*
  小组中每位都有一张卡片
  卡片是6位以内的正整数
  将卡片连起来可以组成多种数字
  计算组成的最大数字

  输入描述：
    ","分割的多个正整数字符串
    不需要考虑非数字异常情况
    小组种最多25个人

   输出描述：
     最大数字字符串

   示例一
     输入
      22,221
     输出
      22221

    示例二
      输入
        4589,101,41425,9999
      输出
        9999458941425101
"""

def auxComp(fisrt,sec):
    v1 = list(fisrt);
    v2 = list(sec);
    len1 = len(v1)
    len2 = len(v2)

    min_len = min(len1,len2) # 相等的情况 不用管，相等的情况已经覆盖
    for ind in range(min_len):
        c1 = v1[ind]
        c2 = v2[ind]
        if c1 != c2 :
            return  int(c2) - int(c1)

    if len1 > len2:
        return v1[0] -v1[min_len]
    else:
        return v2[0] -v2[min_len]

def exam():
   nm= input()
   nums = nm.split(",")

   # sorted(nums,key=cmp_to_key(auxComp))  排序写法不起作用
   print(nums)
   nums.sort(key=cmp_to_key(auxComp))

   res=""
   for  re in nums:
       res = res+ re

   print(res)




if __name__ == '__main__':
    exam()

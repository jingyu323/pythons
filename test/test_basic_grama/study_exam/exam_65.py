from functools import cmp_to_key

""" 
/*
    输入一个由N个大小写字母组成的字符串
    按照ASCII码值从小到大进行排序
    查找字符串中第K个最小ASCII码值的字母(k>=1)
    输出该字母所在字符串中的位置索引(字符串的第一个位置索引为0)
    k如果大于字符串长度则输出最大ASCII码值的字母所在字符串的位置索引
    如果有重复字母则输出字母的最小位置索引

    输入描述
      第一行输入一个由大小写字母组成的字符串
      第二行输入k k必须大于0 k可以大于输入字符串的长度

    输出描述
      输出字符串中第k个最小ASCII码值的字母所在字符串的位置索引
      k如果大于字符串长度则输出最大ASCII码值的字母所在字符串的位置索引
      如果第k个最小ASCII码值的字母存在重复  则输出该字母的最小位置索引

    示例一
     输入
        AbCdeFG
        3
     输出
        5
     说明
       根据ASCII码值排序，第三个ASCII码值的字母为F
       F在字符串中位置索引为5(0为字符串的第一个字母位置索引)

     示例二
       输入
        fAdDAkBbBq
        4
       输出
        6
       说明
        根据ASCII码值排序前4个字母为AABB由于B重复则只取B的第一个最小位置索引6
        而不是第二个B的位置索引8
     */
     
     python 没有三目 运算符
     
"""


def exam():
    line = input()
    k = int(input())
    line_list = list(line)
    line_list.sort()

    if k >= len(line_list):
        res = line_list[len(line_list) - 1]
    else:
        res = line_list[k - 1]
    print(line.index(res))

    line.index()



if __name__ == '__main__':
    exam()

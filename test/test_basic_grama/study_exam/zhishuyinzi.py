# 质数 什么是质数 就是只有 1和它本身是因子 最小的质数是：2
#
import math
num = 180
res=""
# for  ff  in range(2,num):
#     if  num % ff == 0:
#         print("false")
#         res = res + str(ff)+" "
#         num = num /ff
#
#
# i =2
# while i <= num:
#     if  num % i == 0:
#         print("false")
#         res = res + str(i)+" "
#         num = num / i
#     else:
#         i = i +1
# print(res[0:len(res)-1])
# # 只要能除尽说明是因子说明不是质数
# print(10%5)
num1 = int(input())
res=""

for i in range( 2, int(math.sqrt( num1)+1)):
    while num1 % i == 0:
        res = res + str(i)+" "
        num1 = num1 // i
if num1 >1:
    res = res + str(num1) + " "

print(res)
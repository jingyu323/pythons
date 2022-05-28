
# 数组 list
from tensorflow import string

list_data  = [1, 2, 3, 4]

list_data.append(9)

print(list_data)


list_data.insert(4,8)

print(list_data)

list_data[3]=44
print(list_data)

# 多个变量定义 ,总结定义多个变量只需要一个等号 按照顺序赋值就行

a, b, c = 1, 2, "john"
print(a)
print(c)


# 字符串

str="3435ohooret"
print("len is:%d", len(str))

str1='555'
print(int(str1)+88)
str_test="88"+"44"
print(str_test)

## for 循环
for letter in 'Python':  # 第一个实例
    print("当前字母: %s" % letter)

fruits = ['banana', 'apple', 'mango']
for fruit in fruits:  # 第二个实例
    print('当前水果: %s' % fruit)
print("index =====================")
fruits = ['banana', 'apple', 'mango']
for index in range(len(fruits)):
    print('当前水果 : %s' % fruits[index])

print("Good bye!")


print(abs(-4.5))
# while 循环
max_loop = 9
while  max_loop >0:
    if max_loop ==3:

        print( max_loop)
    max_loop =max_loop -1
    if max_loop == 6 :
        print(" this is six")
# if else

# 类型定义
counter = 100 # 直接是整型
miles = 1000.0 # 浮点型
name = "John" # 字符串

# 多个变量赋值
a = b = c = 1
a, b, c = 1, 2, "john"

Python支持四种不同的数字类型：

int（有符号整型）
long（长整型，也可以代表八进制和十六进制）
float（浮点型）
complex（复数）
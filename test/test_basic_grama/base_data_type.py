
# 数组 list
from tensorflow import string

list_data  = [1, 2,44,66,77,0,1, 3, 4]
print(list_data)
list_data.sort()
print(list_data)

list_data.append(9)

print(list_data)

# 列表
vowels = ['e', 'a', 'u', 'o', 'i']
for ch in vowels :
    print(ch)

for index in  range(len(vowels)):
    print("vv index ["+str(index)+"] is :" + vowels[index])

# 降序
vowels.sort(reverse=True)

print(vowels)

list_data.insert(4,8)

print(list_data)

list_data[3]=44
print(list_data)

# 多个变量定义 ,总结定义多个变量只需要一个等号 按照顺序赋值就行

a, b, c = 1, 2, "john"
print(a)
print(c)
num =99
num='{}'.format(num)
print(num)
string = '10'
a = int(string)

# 字符串

str222="3435ohooret"
print("len is:%d", len(str222))

str1='555'
print(int(str1)+88)
str_test="88"+"44"
print(str_test)

## for 循环
for letter in 'Python':  # 第一个实例
    print("当前字母: %s" % letter)

fruits = ['banana', 'apple', 'mango']
# 元素直接遍历
for fruit in fruits:  # 第二个实例
    print('当前水果: %s' % fruit)
print("index =====================")
fruits = ['banana', 'apple', 'mango']
# 按照index 遍历
for index in range(len(fruits)):
    print('当前水果 : %s' % fruits[index])

print("Good bye!")


print(abs(-4.5))
# while 循环
index =0
max_loop = 9
while  max_loop >0:
    if max_loop ==3:

        print( max_loop)
    max_loop =max_loop -1
    if max_loop == 6 :
        print(" this is six")
    index = index +1

    print("indesi:",index)
    print("indesi:%s"%index)
    print("indesi777:{}".format(index) )

 # 逗号拼接的空格
# if else

# 类型定义
counter = 100 # 直接是整型
miles = 1000.0 # 浮点型
name = "John" # 字符串

# 多个变量赋值
a = b = c = 1
a, b, c = 1, 2, "john"

# Python支持四种不同的数字类型：
#
# int（有符号整型）
# long（长整型，也可以代表八进制和十六进制）
# float（浮点型）
# complex（复数）

str111 = 'Hello World!'

print(str111)    # 输出完整字符串
print(str111[0])
  # 输出字符串中的第一个字符
print(str111[2:5])
  # 输出字符串中第三个至第六个之间的字符串
print(str111[2:])  # 输出字符串两次
list = []
list.append(str111[2:])

print(len(list))

print(list)

sites = {'Google', 'Taobao', 'Runoob', 'Facebook', 'Zhihu', 'Baidu'}

num_int = 123
num_str = "456"

print("Data type of num_int:",type(num_int))
print("Data type of num_str:",type(num_str))

print(num_int+int(num_str))
x = str("s1") # x 输出结果为 's1'
y = str(2)    # y 输出结果为 '2'
z = str(3.0)  # z 输出结果为 '3.0'

print(x)

for i in range(9):
    print(i)
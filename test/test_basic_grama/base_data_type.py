
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
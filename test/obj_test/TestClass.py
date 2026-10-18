'''


'''
def reverseWords(input):
    # 通过空格将字符串分隔符，把各个单词分隔为列表
    inputWords = input.split(" ")
    # 翻转字符串
    # 假设列表 list = [1,2,3,4],
    # list[0]=1, list[1]=2 ，而 -1 表示最后一个元素 list[-1]=4 ( 与 list[3]=4 一样)
    # inputWords[-1::-1] 有三个参数
    # 第一个参数 -1 表示最后一个元素
    # 第二个参数为空，表示移动到列表末尾
    # 第三个参数为步长，-1 表示逆向
    inputWords = inputWords[-1::-1]

    # 重新组合字符串
    output = ' '.join(inputWords)
    return output


if __name__ == "__main__":
    input = 'I like runoob'
    rw = reverseWords(input)
    print(rw)

    basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
    print(basket)

    # 创建一个空集合必须用 set() 而不是 { }，因为 { } 是用来创建一个空字典。
    set1 = {1, 2, 3, 4}  # 直接使用大括号创建集合
    set2 = set([4, 5, 6, 7])  # 使用 set() 函数从列表创建集合

    print(set1)
    print(set2)


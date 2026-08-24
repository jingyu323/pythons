

"""
abc 组合在  长字符串中的位置
"""

def exam():
    # str1="abc"
    # str2="efghicabiii"

    str1_in = input()

    str1 =str1_in.split()[0]
    str2 = str1_in.split()[1]
    str1_list = list(str1)
    str1_list.sort()
    min_index_list = [];

    for ch_index in range(len(str1_list)):
        find_index = str2.index(str1_list[ch_index])
        if find_index >= 0:
            if find_index + len(str1) > len(str2):
                find_index = -1
                break
            com_str = str2[find_index:find_index + len(str1)]
            com_str_list = list(com_str)
            com_str_list.sort()
            # 找到之后遍历
            for com_index in range(len(str1_list)):
                if com_str_list[com_index] != str1_list[com_index]:
                    find_index = -1
                    break
        if find_index > 0:
            min_index_list.append(find_index)

    min_res = 1000000
    for index in range(len(min_index_list)):
        min_res = min(min_res, min_index_list[index])
    print(min_res)


if __name__ == '__main__':
    res = exam()
    print(res)

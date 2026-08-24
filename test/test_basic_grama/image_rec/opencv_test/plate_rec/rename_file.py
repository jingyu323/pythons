import os

template = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
            'X', 'Y', 'Z',
            '藏', '川', '鄂', '甘', '赣', '贵', '桂', '黑', '沪', '吉', '冀', '津', '晋', '京', '辽', '鲁', '蒙',
            '闽', '宁',
            '青', '琼', '陕', '苏', '皖', '湘', '新', '渝', '豫', '粤', '云', '浙']

def read_directory(directory_name):
    referImg_list = []
    for filename in os.listdir(directory_name):
        referImg_list.append(directory_name + "/" + filename)
    return referImg_list
def get_chinese_words_list(template):
    chinese_words_list = []
    for i in range(34, 64):
        # 将模板存放在字典中
        c_word = read_directory('./refer1/' + template[i])
        if len(c_word) > 0:

            chinese_words_list.append(c_word)
    return chinese_words_list

chinese_words_list = get_chinese_words_list(template)

print(chinese_words_list)

for chinese_words in chinese_words_list:
    for wordd in chinese_words:

        if wordd.find("ԥ_") >=0:
            print(wordd.replace("ԥ_","豫_"))
            os.rename(wordd, wordd.replace("��_","津_"))
        if wordd.find("��_") >=0:
            print(wordd.replace("��_","津_"))
            os.rename(wordd, wordd.replace("��_","津_"))
        if wordd.find("³_") >=0:
            print(wordd.replace("³_","鲁_"))
            os.rename(wordd, wordd.replace("³_","鲁_"))
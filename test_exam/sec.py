from functools import cmp_to_key
start=int(input())
total=int(input())
def com(wo1,wo2):
    return len(wo1) -len(wo2)

intput_list = []
for i in  range(total):
    wd=input()
    intput_list.append(wd)

intput_list.sort(key=com)

print(intput_list)

fisst_word = intput_list[start]
intput_list.remove(fisst_word)



print(fisst_word)


"""

单词接龙 

如果长度相同按照字典顺序排序取最小
参加过接龙的单词不能再次参加

最后一位为下一个单词的首字母

0 为选中的第一个单词
6 为单词数组的个树
word
dd
da
dc
dword
d

wordwordda


"""



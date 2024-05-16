"""
幼儿园两个班的小朋友在排队时混在了一起，每位小朋友都知道自己是否与前面一位小朋友是否同班，请你帮忙把同班的小朋友找出来。
小朋友的编号为整数，与前一位小朋友同班用Y表示，不同班用N表示。

题目要求
输入为：空格分开的小朋友编号和是否同班标志。
比如：6/N 2/Y 3/N 4/Y，表示共4位小朋友，2和6同班，3和2不同班，4和3同班。
其中，小朋友总数不超过999，每个小朋友编号大于0，小于等于999。
不考虑输入格式错误问题。

输出为：两行，每一行记录一个班小朋友的编号，编号用空格分开。且：

编号需要按照大小升序排列，分班记录中第一个编号小的排在第一行。
若只有一个班的小朋友，第二行为空行。
若输入不符合要求，则直接输出字符串ERROR。
示例1
输入
1/N 2/Y 3/N 4/Y
输出
1 2
3 4
说明
2的同班标记为Y，因此和1同班。

"""
def exam():

    fenban="1/N 2/Y 3/N 4/Y"
    fenban_arr =fenban.split()

    fiest_class=[]
    sec_class=[]

    fiest_class.append(fenban_arr[0].split("/")[0])
    pre_class = "jia"
    for fb in  range(1,len(fenban_arr)):
        flag = fenban_arr[fb].split("/")[1]
        if flag == "Y":
            if pre_class == "jia" :
                fiest_class.append(fenban_arr[fb].split("/")[0])
                pre_class = "jia";
            else:
                sec_class.append(fenban_arr[fb].split("/")[0])
                pre_class = "yi";
        else:
            if pre_class == "jia" :
                sec_class.append(fenban_arr[fb].split("/")[0])
                pre_class = "yi";
            else:
                fiest_class.append(fenban_arr[fb].split("/")[0])
                pre_class = "jia";




    fiest_class.sort()
    sec_class.sort()

    print(fiest_class)
    print(sec_class)



if __name__ == '__main__':
    exam()

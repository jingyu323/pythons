"""
  在学校中
  N个小朋友站成一队
  第i个小朋友的身高为height[i]
  第i个小朋友可以看到第一个比自己身高更高的小朋友j
  那么j是i的好朋友
  (要求：j>i)
  请重新生成一个列表
  对应位置的输出是每个小朋友的好朋友的位置
  如果没有看到好朋友
  请在该位置用0代替
  小朋友人数范围 0~40000

  输入描述：
    第一行输入N
    N表示有N个小朋友

    第二行输入N个小朋友的身高height[i]
    都是整数

  输出描述：
    输出N个小朋友的好朋友的位置

  示例1：
     输入：
       2
       100 95
      输出
       0 0
     说明
       第一个小朋友身高100站在队伍末尾
       向队首看 没有比他身高高的小朋友
       所以输出第一个值为0
       第二个小朋友站在队首前面也没有比他身高高的小朋友
       所以输出第二个值为0

   示例2：
      输入
        8
        123 124 125 121 119 122 126 123
      输出
        1 2 6 5 5 6 0 0
       说明：
       123的好朋友是1位置上的124
       124的好朋友是2位置上的125
       125的好朋友是6位置上的126
        依此类推


"""

def exam():
    first = input("first:")
    first = int(first)

    if first == 0:
        print(first)
        return

    line_res = input("input result:")
    print(line_res)
    strs = line_res.split(" ")

    print(strs)
    res = []
    for index_out in range(len(strs)):
        pos = 0;
        for index_in  in range(index_out +1,len(strs)):
             if int(strs[index_in]) > int(strs[index_out]) :
                 pos = index_in
                 break
        res.append(pos)

    builder = ""
    for re in res:
        builder = builder + str(re) + " "

    if len(builder) > 1:
         substring = builder[0:len(builder)-1]
         print(substring)



if __name__ == '__main__':
    exam()
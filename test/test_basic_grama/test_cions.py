
# 题目
# 题目描述
# 在一个国家仅有1分，2分，3分硬币，将钱N分兑换成硬币有很多种兑法。请你编程序计算出共有多少种兑法。
#
# 解答要求
# 时间限制：1000ms, 内存限制：64MB
# 输入
# 输入每行包含一个正整数N(0<N<32768)。输入到文件末尾结束。
#
# 输出
# 输出对应的兑换方法数。
#
# 样例
# 输入样例 1 复制
#
# 3
# 2934
# 输出样例 1
#
# 3
# 718831

def change_cion():
    from pip._vendor.distlib.compat import raw_input
    n = int( raw_input('input a number:'))
    sum = 0;
    threee_max = n//3 +1
    for i in range(threee_max):
        sum += (n - i * 3) // 2 + 1;
    print(sum)


if __name__ == '__main__':
    change_cion()
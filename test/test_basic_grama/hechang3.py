import bisect  # 导入查找模块
""""
https://www.nowcoder.com/practice/6d9d69e3898f45169a441632b325c7b4
"""

def get_max_sub(arr):  # 定义获取最长子序列函数
    res = [arr[0]]  # 将传入的列表第一个参数放入res
    dp = [1] * len(arr)  # 定义一个长度为输入列表长度的列表，元素为1.
    for i in range(1, len(arr)):  # 计算以arr[i]结尾的最长上升子序列长度
        if arr[i] > res[-1]:  # 如果arr[i]大于最后一个元素，插入
            res.append(arr[i])
            dp[i] = len(res)
        else:  # 如果arr[i]小于最后一个元素，找到res中比他大的元素的位置，并将该元素替换为arr[i]
            index = bisect.bisect_left(res, arr[i])
            res[index] = arr[i]
            dp[i] = index + 1
    return dp





while True:
    try:
        n = int(input())
        lst = list(map(int, input().split()))
        left = get_max_sub(lst)  # 最长升序子序列
        right = get_max_sub(lst[::-1])[::-1]  # 最长降序子序列
        ans = [left[i] + right[i] - 1  for i in range(len(lst))]   # 每个元素多计算1次，减去
        print(n - max(ans))  # 注意是n - 最长子序列，即为剔除人数
    except:
        break
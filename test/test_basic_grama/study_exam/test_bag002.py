def test_2_wei_bag_problem1(bag_size, weight, value) -> int:
    rows, cols = len(weight), bag_size + 1
    # 创建初始化数组
    dp = [[0 for _ in range(cols)] for _ in range(rows)]

    for i in range(rows):

        dp[i][0] = 0
        print(dp[i])
    first_item_weight, first_item_value = weight[0], value[0]
# 初始化第一行数组，其实也就是初始化  动态规划的起始值，先初始化列，后在初始化第一行的值，因为这个是可以直接确定的
    for j in range(1, cols):
        if first_item_weight <= j:
            dp[0][j] = first_item_value
    # 更新dp数组: 先遍历物品, 再遍历背包.
    for i in range(1, len(weight)):
        cur_weight, cur_val = weight[i], value[i]
        for j in range(1, cols):
            if cur_weight > j:  # 说明背包装不下当前物品.
                dp[i][j] = dp[i - 1][j]  # 所以不装当前物品.
            else:
                # 定义dp数组: dp[i][j] 前i个物品里，放进容量为j的背包，价值总和最大是多少。
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - cur_weight] + cur_val)
    print(dp)

if __name__ == "__main__":
	bag_size = 10
	weight = [2,2,6,5,4]
	value = [6,3,5,4,6]
	test_2_wei_bag_problem1(bag_size, weight, value)
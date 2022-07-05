def test_2_wei_bag_problem1(bag_size, weight, value) -> int:
    rows, cols = len(weight), bag_size + 1
    dp = [[0 for _ in range(cols)] for _ in range(rows)]

    for i in range(rows):
        print(dp[i][0])
        dp[i][0] = 0
    first_item_weight, first_item_value = weight[0], value[0]
# 初始化第一行数组
    for j in range(1, cols):
        if first_item_weight <= j:
            dp[0][j] = first_item_value
    print(dp)

if __name__ == "__main__":
	bag_size = 10
	weight = [2,2,6,5,4]
	value = [6,3,5,4,6]
	test_2_wei_bag_problem1(bag_size, weight, value)
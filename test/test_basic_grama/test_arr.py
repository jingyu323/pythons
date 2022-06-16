import bisect

list=[1,2,3,4]

tem = list


tem[0]=3

print(list)


dp = [1] * len(list)

print(dp)

index = bisect.bisect_left(res, arr[i])
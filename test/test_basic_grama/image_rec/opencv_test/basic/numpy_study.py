import random
from datetime import time


from timeit import timeit

import numpy as np
from matplotlib import pyplot as plt


def demo1_np():
    # score = np.array([[12,3 ],[3,5 ]] )
    # print(score)

    a = []
    for i in range(10000):
        a.append(random.random( ))

    sum1 = sum(a)
    print(sum1)

    x2 = np.random.normal(1.75,1,10000000)

    plt.hist(x2,10000)
    plt.show()

    arr= np.array(a)
    #  去重
    print(np.unique(arr))


"""
astype（np.int32） 修改类型

np.unique() 去重

"""

# 函数执行逐元素的按位与操作。按位与操作是位运算的一种，主要用于将两个整数在相应位上进行比较，只有在对应位都为1时，结果才为1，否则结果为0
"""
1 & 4 = 0001 & 0100 = 0000 -> 0
2 & 3 = 0010 & 0011 = 0010 -> 2
"""

def bitwise_and_test():
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    b = np.array([4, 3, 2, 1], dtype=np.int32)
    result = np.bitwise_and(a, b)
    print(result)



def ones_demo():
    a = np.ones((10,10))
    print(a)
    print(a.dtype)





if __name__ == '__main__':
    ones_demo()

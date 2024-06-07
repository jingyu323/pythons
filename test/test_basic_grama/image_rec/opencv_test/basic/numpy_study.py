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

if __name__ == '__main__':
    demo1_np()

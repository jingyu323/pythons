def number (x):
    for i in range(2,x):
        if x%i==0:
            print("这个数不是质数")
            break
    else:
        print("这个数是质数")




""""



判断 素数 只要有一个因子就不是素数

"""
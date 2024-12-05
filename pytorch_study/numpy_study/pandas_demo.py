from pandas import DataFrame


def demo1():
    df01 = DataFrame([[1, 2], [3, 4]],index=["one","two"],columns=['a', 'b'])
    print(df01)

    print(df01.mean())
    print(df01.sum())
    print(df01.describe())



if __name__ == '__main__':

    demo1()
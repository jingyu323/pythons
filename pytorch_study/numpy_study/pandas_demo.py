import pandas as pd
from pandas import DataFrame


def demo1():
    df01 = DataFrame([[1, 2], [3, 4]], index=["one", "two"], columns=['a', 'b'])
    print(df01)

    print(df01.mean())
    print(df01.sum())
    print(df01.describe())
    # 字典的结构如下，字典键值是列名，字典的值是每一列数据
    dic = {
        'name': ['张三', '李四', '王二'],
        'num': [802, 807, 801],
        'height': [183, 161, 163],
        'weight': [87, 60, 71],
        'gender': ['男', '男', '男'],
        'age': [25, 30, 25]
    }
    df = pd.DataFrame(dic)
    print(df)

    # 列表每一个元素是字典，获取的数据类型经常是列表，需要把列表转成dataframe
    lis = [{'id': 1, 'num': 4, '年龄': 15},
           {'id': 2, 'num': 6, '年龄': 155},
           {'id': 3, 'num': 8, '年龄': 415}]

    print(df)

    df = pd.DataFrame(lis)
    print(df.empty)  # 打印出df是False
    # 行数
    print('行数：%s' % df.shape[0])
    # 列数
    print('列数：%s' % df.shape[1])
    df = pd.DataFrame()
    print(df.empty)  # 打印出df是Ture


def demo2():
    lis = [{'id': 1, 'num': 4, '年龄': 15}, {'id': 2, 'num': 6, '年龄': 155}]
    df = pd.DataFrame(lis)
    # 展示df
    print('更改前：')
    print(df)

    # 更改列名df.columns = [‘A’,‘B’]
    df.columns = ['工号', '数字', '代号']
    # 展示df
    print("df.columns = ['A','B']更改后：")
    print(df)

    # 更改列名​df.rename(columns={‘a’:‘A’})
    df.rename(columns={'工号': '员工号'}, inplace=True)
    print("df.rename(columns={'a':'A'})更改后：")
    print(df)


"""
df["年龄"] = pd.to_numeric(df["年龄"], errors='coerce')  # 将某列转换成数字类型

        df['年龄'] = df['年龄'].astype('int')  # 把列名为"年龄"的数据类型转换成int类型
"""


def demo3():
    lis = [{'id': 1, 'num': 4, '年龄': '15'}, {'id': 2, 'num': 6, '年龄': '155'}]
    df = pd.DataFrame(lis)
    df.reset_index(drop=True, inplace=True)
    df = pd.DataFrame(df)

    df['年龄'] = df['年龄'].astype('int')  # 把列名为"年龄"的数据类型转换成int类型
    df['年龄'] = df['年龄'].astype('float')  # 把列名为"年龄"的数据类型转换成float类型
    df['年龄'] = df['年龄'].astype('str')  # 把列名为"年龄"的数据类型转换成str类型
    df[['年龄', 'id']] = df[['年龄', 'id']].astype('str')  # 把列名为"年龄"、"id"的数据类型转换成str类型
    df["年龄"] = pd.to_numeric(df["年龄"], errors='coerce')  # 将某列转换类型
    print(df.dtypes)  # 查看所有列的数据类型
    print(df['年龄'].dtypes)

    print(df)
    df['age'] = 1  # 增加一列，列名age，每个值为1
    df.loc[3, :] = 1  # 索引3的这行所有数据改成1，没有就增加，有就更改
    print(df)

    dic = {
        'name': ['张三', '李四', '王二', '麻子', '小红', '小兰', '小玉', '小强', '小娟', '小明'],
        'num': [802, 807, 801, 803, 806, 805, 808, 809, 800, 804],
        'height': [183, 161, 163, 163, 156, 186, 184, 154, 153, 174],
        'weight': [87, 60, 71, 74, 45, 50, 47, 67, 49, 70],
        'gender': ['男', '男', '男', '男', '女', '女', '女', '男', '女', '男'],
        'age': [25, 30, 25, 26, 27, 20, 23, 26, 30, 30]
    }
    df = pd.DataFrame(dic)  # 转成dataframe

    df = df[df['name'].str.contains('三')]  # contains用法，name包含三的数据
    print("", df)

    df = df[df['name'].isin(['张三'])]  # isin用法，name是张三的数据
    df = df[~df['name'].isin(['张三'])]
    print(df)


if __name__ == '__main__':
    demo2()
    demo3()

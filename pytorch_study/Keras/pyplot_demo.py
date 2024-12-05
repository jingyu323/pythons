import numpy as np

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def demo1():
    # 使用NumPy生成数据
    x = np.linspace(0, 10, 100)  # 0到10之间的100个等间距数
    y = np.sin(x)  # 每个x对应的正弦值

    # 使用plt.plot()绘制图形
    plt.plot(x, y,color='red', linewidth=2, linestyle='--', marker='o', markersize=10,label='数据1' )  # 绘制x和y对应的点，并连接它们形成线条

    # 添加标题和轴标签
    plt.title('Sine Wave')
    plt.xlabel('x')
    plt.ylabel('y')
    # 添加图例
    plt.legend()
    # 显示网格线和设置背景色
    plt.grid(True)
    plt.gca().set_facecolor('lightgrey')
    # 显示图形
    plt.show()


def demo2():
    font = fm.FontProperties(size=22)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    x = [1, 2, 3, 4, 5]
    y = [10, 15, 13, 18, 16]

    # 绘制线图，并自定义外观
    plt.plot(
        x,  # X轴数据
        y,  # Y轴数据
        marker='o',  # 标记样式：圆点
        linestyle='-',  # 线条样式：实线
        color='green',  # 线条颜色：蓝色
        linewidth=2,  # 线宽：2
        markersize=10,  # 标记大小：8
        label='数据2'
    )


    # 添加标签和标题
    plt.xlabel('X轴标签')
    plt.ylabel('Y轴标签')
    plt.title('标题')

    # 添加图例
    plt.legend()
    # 显示网格线
    plt.grid(True)
    # 自定义刻度
    plt.xticks([1, 2, 3, 4, 5], [ '2', '3', '4', '6', '8'], fontproperties=font)
    # plt.xticks([1, 2, 3, 4, 5], ['11', '22', '33', '44', '55'],rotation=60,fontsize=22)
    # 显示图表
    plt.show()

def demo3():
    import matplotlib.pyplot as plt
    # 显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 示例数据

    x = [1, 2, 3, 4, 5]
    y = [10, 15, 13, 18, 16]

    # 绘制线图，并自定义外观
    plt.plot(
        x,  # X轴数据
        y,  # Y轴数据
        marker='o',  # 标记样式：圆点
        linestyle='-',  # 线条样式：实线
        color='green',  # 线条颜色：蓝色
        linewidth=2,  # 线宽：2
        markersize=10,  # 标记大小：8
        label='数据1'  # 图例标签
    )

    # 添加标签和标题
    plt.xlabel('X轴标签')
    plt.ylabel('Y轴标签')
    plt.title('标题')

    # 添加图例
    plt.legend()

    # 显示网格线
    plt.grid(True)

    # 自定义刻度
    plt.xticks([1, 2, 3, 4, 5], ['一', '二', '三', '四', '五'])

    # 显示图表
    plt.show()


if __name__ == '__main__':
    demo3()
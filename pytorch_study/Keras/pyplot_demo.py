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

def demo4():
    category_names = ['Strongly disagree', 'Disagree',
                      'Neither agree nor disagree', 'Agree', 'Strongly agree']
    results = {
        'Question 1': [10, 15, 17, 32, 26],
        'Question 2': [26, 22, 29, 10, 13],
        'Question 3': [35, 37, 7, 2, 19],
        'Question 4': [32, 11, 9, 15, 33],
        'Question 5': [21, 29, 5, 5, 40],
        'Question 6': [8, 19, 5, 30, 38]
    }

    def survey(results, category_names):
        """
        Parameters
        ----------
        results : dict
            A mapping from question labels to a list of answers per category.
            It is assumed all lists contain the same number of entries and that
            it matches the length of *category_names*.
        category_names : list of str
            The category labels.
        """
        labels = list(results.keys())
        data = np.array(list(results.values()))
        data_cum = data.cumsum(axis=1)
        category_colors = plt.colormaps['RdYlGn'](
            np.linspace(0.15, 0.85, data.shape[1]))

        print(category_colors)

        fig, ax = plt.subplots(figsize=(9.2, 5))
        ax.invert_yaxis()
        ax.xaxis.set_visible(False)
        ax.set_xlim(0, np.sum(data, axis=1).max())

        for i, (colname, color) in enumerate(zip(category_names, category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths
            rects = ax.barh(labels, widths, left=starts, height=0.5,
                            label=colname, color=color)

            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            ax.bar_label(rects, label_type='center', color=text_color)
        ax.legend(ncols=len(category_names), bbox_to_anchor=(0, 1),
                  loc='lower left', fontsize='small')

        return fig, ax

    survey(results, category_names)
    plt.show()


def demo_sublot():
    x = np.linspace(0, 2 * np.pi, 400)
    y = np.sin(x ** 2)

    # Create just a figure and only one subplot
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title('Simple plot')
    # Create two subplots and unpack the output array immediately
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(x, y)
    ax1.set_title('Sharing Y axis')
    ax2.scatter(x, y)

    # Create four polar Axes and access them through the returned array
    fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection="polar"))
    axs[0, 0].plot(x, y)
    axs[1, 1].scatter(x, y)

    # Share a X axis with each column of subplots
    plt.subplots(2, 2, sharex='col')

    # Share a Y axis with each row of subplots
    plt.subplots(2, 2, sharey='row')

    # Share both X and Y axes with all subplots
    plt.subplots(2, 2, sharex='all', sharey='all')

    # Note that this is the same as
    plt.subplots(2, 2, sharex=True, sharey=True)

    # Create figure number 10 with a single subplot
    # and clears it if it already exists.
    fig, ax = plt.subplots(num=10, clear=True)
    plt.show()

if __name__ == '__main__':
    demo4()
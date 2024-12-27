import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl


def  plot_demo():

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [1, 4, 2, 3, 5, 2, 9, 5, 8, 6]
    plt.plot(x, y)  # 展示的数据
    plt.savefig('demo.png')
    plt.show()

def plot_draw():
    plt.figure()
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [1, 4, 2, 3, 5, 2, 9, 5, 8, 6]

    y2 = [3, 6, 5, 2, 8, 7, 2, 3, 9, 6]


    plt.plot(x, y,label='FS')
    plt.plot(x, y2, color='r',linestyle='--',label='SH')

    plt.grid(True,linestyle='--',alpha=0.2)

    plt.xlabel("时间")
    plt.ylabel("大小")
    plt.title("测试 title" )
    plt.legend(loc="best")
    plt.show()
def plot_draw_sub():
    plt.figure()
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [1, 4, 2, 3, 5, 2, 9, 5, 8, 6]

    y2 = [3, 6, 5, 2, 8, 7, 2, 3, 9, 6]

    fig,axes= plt.subplots(1,2 ,figsize = (20,5),dpi=100)


    axes[0].plot(x, y,label='FS',linestyle='-.')
    axes[1].plot(x, y2, color='r',linestyle='--',label='SH')

    axes[0].grid(True,linestyle='--',alpha=0.2)
    axes[1].grid(True,linestyle='-.',alpha=0.2)

    axes[0].set_xticks(x )
    axes[0].set_yticks(y )
    axes[0].set_xticklabels(x)

    axes[1].set_xticks(x )
    axes[1].set_yticks(y2 )
    axes[1].set_xticklabels(x)
    # axes[1].set_xticks(x )
    # axes[1].set_yticks(y2)
    # axes[1].set_xticks(xlable[::2])
    # axes[1].set_yticks(ylable[::5])
    # axes[1].title("测试 title1111" )
    # axes[0].title("测试 title0000" )


    axes[0].set_xlabel("时间")
    axes[0].set_ylabel("大小")
    axes[1].set_xlabel("时间1")
    axes[1].set_ylabel("大小1")

    axes[1].set_title("测试 title1")
    axes[0].set_title("测试 title0")
    # 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    axes[0].legend(loc="upper left" )
    axes[1].legend( loc="upper left")
    plt.show()

def plot_draw_math():
    x= np.linspace(-10,10,1000)
    y = np.sin(x)

    plt.figure()
    plt.plot(x,y)
    plt.grid(True,linestyle='--',alpha=0.2)
    plt.show()

if __name__ == '__main__':
    plot_draw_math()
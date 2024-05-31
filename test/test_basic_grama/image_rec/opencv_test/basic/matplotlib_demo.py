import matplotlib.pyplot as plt



def  plot_demo():

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [1, 4, 2, 3, 5, 2, 9, 5, 8, 6]
    plt.plot(x, y)  # 展示的数据
    plt.savefig('demo.png')
    plt.show()


if __name__ == '__main__':
    plot_demo()
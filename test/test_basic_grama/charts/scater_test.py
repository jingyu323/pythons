from pyecharts.faker import Faker
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Scatter

from pyecharts.charts import Scatter3D
# from pyecharts.render import notebook
import random



def test_scatter1():
    # 数据
    x_data = [1, 2, 3, 4, 5]
    y_data = [10, 20, 30, 40, 50]

    # 创建Scatter实例
    scatter = Scatter().set_global_opts(
        title_opts=opts.TitleOpts(title="普通气泡图")
    )

    # 添加数据系列
    scatter.add_xaxis(x_data)
    scatter.add_yaxis("", y_data)

    # 设置气泡的大小和颜色
    scatter.set_series_opts(
        label_opts=opts.LabelOpts(is_show=False),
        itemstyle_opts=opts.ItemStyleOpts(color="rgba(255, 0, 0, 0.6)"),
        symbol_size=20
    )

    # 渲染图表
    scatter.render_notebook()
    scatter.render("scatter.html")


def  mul_scater():

    # 创建Scatter对象
    scatter = (
        Scatter()
            # 添加x轴数据
            .add_xaxis(Faker.choose())
            # 添加y轴数据，系列名称为"商家A"
            .add_yaxis("商家A", Faker.values())
            # 添加y轴数据，系列名称为"商家B"
            .add_yaxis("商家B", Faker.values())
            # 设置全局配置项
            .set_global_opts(
            # 设置标题
            title_opts=opts.TitleOpts(title="多维度散点图"),
            # 设置视觉映射配置项，类型为"size"，最大值为150，最小值为20
            visualmap_opts=opts.VisualMapOpts(type_="size", max_=150, min_=20),
        )
    )

    # 在Jupyter Notebook中渲染图表
    scatter.render_notebook()
    scatter.render("mulscatter.html")

def scater_with_line():


    # 生成随机数据
    np.random.seed(0)
    data = np.random.randn(100, 4)

    # 使用链式写法绘制散点图
    scatter = (
        Scatter()
            .add_xaxis(list(range(100)))
            .add_yaxis("A", data[:, 0].tolist())
            .add_yaxis("B", data[:, 1].tolist())
            .add_yaxis("C", data[:, 2].tolist())
            .add_yaxis("D", data[:, 3].tolist())
            .set_global_opts(
            title_opts=opts.TitleOpts(title="多维度散点图"),
            xaxis_opts=opts.AxisOpts(name="Index"),
            yaxis_opts=opts.AxisOpts(name="Value"),
        )
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    )

    # 添加分割线
    line = (
        Scatter()
            .add_xaxis(list(range(100)))
            .add_yaxis("Line", [0] * 100, label_opts=opts.LabelOpts(is_show=False))
            .set_series_opts(
            symbol_size=0,
            linestyle_opts=opts.LineStyleOpts(type_="dashed", color="red"),
            label_opts=opts.LabelOpts(is_show=False),
        )
    )

    scatter.render_notebook()
    scatter.render("scater_with_line.html")

def scater_lake():
    # 导入所需的类和函数
    from pyecharts import options as opts
    from pyecharts.charts import EffectScatter
    from pyecharts.faker import Faker

    # 创建 EffectScatter 对象，并设置 x 轴数据和 y 轴数据
    c = (
        EffectScatter()
            .add_xaxis(Faker.choose())  # 添加 x 轴数据，这里使用了 Faker.choose() 生成随机数据
            .add_yaxis("", Faker.values())  # 添加 y 轴数据，这里使用了 Faker.values() 生成随机数据
            .set_global_opts(title_opts=opts.TitleOpts(title="动态涟漪散点图"))  # 设置全局配置，标题为"动态涟漪散点图"
    )

    # 在 Jupyter Notebook 中展示
    c.render_notebook()
    c.render("scater_lake.html")

def diff_shape_scater():
    # 创建散点图对象
    scatter = Scatter()

    # 设置图表标题和大小
    scatter.set_global_opts(
        title_opts=opts.TitleOpts(title="不同形状散点图"),
        visualmap_opts=opts.VisualMapOpts(type_="size", max_=50, min_=20),
        legend_opts=opts.LegendOpts(pos_right="10%", pos_bottom="15%"),
    )

    # 添加散点图数据并设置样式
    scatter.add_xaxis(['A', 'B', 'C', 'D', 'E'])
    scatter.add_yaxis("圆形", [10, 20, 30, 40, 50], symbol_size=10)
    scatter.add_yaxis("矩形", [20, 30, 40, 50, 60], symbol='rect', symbol_size=15)
    scatter.add_yaxis("三角形", [30, 40, 50, 60, 70], symbol='triangle', symbol_size=20)
    scatter.add_yaxis("星形", [40, 50, 60, 70, 80], symbol='star', symbol_size=25)

    # 在 Jupyter Notebook 中展示
    scatter.render_notebook()
    scatter.render("diff_shape_scater.html")


def  threed_scater():

    scatter_data = [[random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)] for _ in range(80)]

    scatter3d = (
        Scatter3D()
            .add("", scatter_data)
            .set_global_opts(
            title_opts=opts.TitleOpts(title="3D散点图"),
            visualmap_opts=opts.VisualMapOpts(max_=10),
        )
    )

    scatter3d.render_notebook()
    scatter3d.render("threed_scater.html")


if __name__ == "__main__" :
    threed_scater()
from pyecharts.globals import ThemeType
from pyecharts.charts import Bar3D
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.faker import Faker
def  bar_custerm_bg():


    # 定义x轴的数据
    x_data = ["语文", "数学", "英语", "历史", "政治", "地理", "体育"]
    # 定义学生A的成绩数据
    y_value1 = [99, 85, 90, 95, 70, 92, 98]
    # 定义学生B的成绩数据
    y_value2 = [90, 95, 88, 85, 96, 80, 76]
    add_value2 = [10, 10, 10, 10, 10, 10]  # 学生B是三好学生每门课程加10分

    # 创建柱状图实例
    bar = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.CHALK))  # 使用CHALK主题
        .add_xaxis(x_data)  # 添加x轴数据
        .add_yaxis("学生A", y_value1)  # 添加学生A的成绩数据，设置系列名称为"学生A"
        .add_yaxis("学生B", y_value2, stack="stack1")  # 添加学生B的成绩数据，设置系列名称为"学生B"
        .add_yaxis("加分", add_value2, stack="stack1")  # 添加数据

        .set_series_opts(
            title_opts=opts.TitleOpts(title="我是主标题"),
            xaxis_opts=opts.AxisOpts(name="课程类别"),  # 添加x坐标轴名称
            yaxis_opts=opts.AxisOpts(name="课程成绩"),  # 添加y坐标轴名称
            label_opts=opts.LabelOpts(is_show=False),
            markline_opts=opts.MarkLineOpts(
                data=[
                    opts.MarkLineItem(type_="min", name="最小值"),
                    opts.MarkLineItem(type_="max", name="最大值"),
                    opts.MarkLineItem(type_="average", name="平均值"),
                ]
            ),
        )
    )

    # 设置全局配置项
    bar.set_global_opts(
        # 添加自定义背景图
        graphic_opts=[
            opts.GraphicImage(
                graphic_item=opts.GraphicItem(
                    id_="background",
                    right=20,
                    bottom=20,
                    z=-10,
                    bounding="raw",
                    origin=[75, 75],
                ),
                graphic_imagestyle_opts=opts.GraphicImageStyleOpts(
                    image="https://kr.shanghai-jiuxin.com/file/bizhi/20220927/4gzitkl1lyv.jpg", #  指定一下路径。也可以用文件路径
                    width=650,
                    height=450,
                    opacity=0.3,
                ),
            )
        ],
    )

    # 在jupyter notebook中渲染图表
    bar.render_notebook()
    bar.render("bar_custerm_bg.html")


def  threed_bar():


    # 定义x、y、z轴的数据
    x_data = ["A", "B", "C", "D", "E"]
    y_data = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    z_data = [
        [1, 2, 3, 4, 5, 6, 7],
        [2, 3, 4, 5, 6, 7, 8],
        [3, 4, 5, 6, 7, 8, 9],
        [4, 5, 6, 7, 8, 9, 10],
        [5, 6, 7, 8, 9, 10, 11]
    ]

    # 创建三维柱状图实例
    bar3d = (
        Bar3D()  # 使用Bar3D()函数创建一个Bar3D对象
            .add(
            "",
            [[i, j, z_data[i][j]] for i in range(len(x_data)) for j in range(len(y_data))],
            xaxis3d_opts=opts.Axis3DOpts(data=x_data, type_="category"),
            yaxis3d_opts=opts.Axis3DOpts(data=y_data, type_="category"),
            zaxis3d_opts=opts.Axis3DOpts(type_="value"),
            grid3d_opts=opts.Grid3DOpts(width=200, height=80, depth=80),
        )
            .set_global_opts(
            visualmap_opts=opts.VisualMapOpts(max_=11),
            title_opts=opts.TitleOpts(title="三维柱状图"),
        )
    )

    # 在jupyter notebook中渲染图表
    bar3d.render_notebook()
    bar3d.render("threed_bar.html")

def  scale_bar():


    # 创建柱状图实例
    bar = (
        Bar()  # 使用Bar()函数创建一个Bar对象
            .add_xaxis(Faker.days_attrs)  # 添加x轴数据
            .add_yaxis("出售量", Faker.days_values, color=Faker.rand_color())  # 添加y轴数据，设置系列名称为"出售量"，颜色随机生成
            .set_global_opts(
            title_opts=opts.TitleOpts(title="水平滑动、鼠标滚轮缩放柱状图"),  # 设置标题
            datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],  # 设置DataZoom组件，包括滑动条和内置型
        )
    )

    # 在jupyter notebook中渲染图表
    bar.render_notebook()
    bar.render("scale_bar.html")


if __name__ == "__main__" :
    scale_bar()
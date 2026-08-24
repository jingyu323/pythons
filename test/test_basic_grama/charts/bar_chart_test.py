from pyecharts.globals import ThemeType
from pyecharts.charts import Bar3D, Geo, Map
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

def  geo_heatmap():
    geo = Geo(init_opts=opts.InitOpts(theme='light',
                                      width='1000px',
                                      height='600px'))

    geo.add_schema(maptype="china")

    geo.add("",
            [("广州", 150), ("成都", 70), ("南昌", 64), ("苏州", 100), ("郑州", 63)],
            type_='heatmap')
    # 热点图必须配置visualmap_opts
    geo.set_global_opts(visualmap_opts=opts.VisualMapOpts())


    geo.render_notebook()
    geo.render()
    geo.render("geo_heatmap.html")




data = [('广东省', 129118.58),
            ('山东省', 87435),
            ('河南省', 61345),
            ('四川省', 56749.80),
            ('江苏省', 122875.60),
            ('河北省', 42370.40),
            ('湖南省', 48670.37),
            ('安徽省', 45045),
            ('湖北省', 53734.92),
            ('浙江省', 77715),
            ('广西壮族自治区', 26300.87),
            ('云南省', 28954.20),
            ('江西省', 32074.7),
            ('辽宁省', 28975.1),
            ('黑龙江省', 15901),
            ('陕西省', 32772.68),
            ('山西省', 25642.59),
            ('福建省', 53109.85),
            ('贵州省', 20164.58),
            ('重庆市', 29129.03),
            ('吉林省', 13070.24),
            ('甘肃省', 11201.60),
            ('内蒙古自治区', 23159),
            ('台湾省', 51262.8),
            ('上海市', 44652.8),
            ('新疆维吾尔自治区', 17741.34),
            ('北京市', 41610.9),
            ('天津市', 16311.34),
            ('海南省', 6818.22),
            ('香港特别行政区', 23740),
            ('宁夏回族自治区', 5069.57),
            ('青海省', 3610.1),
            ('西藏自治区', 2134.62),
            ('澳门特别行政区', 1929.27)]

def map_with_viusalmap():
        map_chart = Map(init_opts=opts.InitOpts(theme='light',
                                                width='1000px',
                                                height='600px'))
        map_chart.add('GDP（亿人民币）',
                      data_pair=data,
                      maptype='china',
                      # 关闭symbol的显示
                      is_map_symbol_show=False)

        map_chart.set_global_opts(visualmap_opts=opts.VisualMapOpts(
            max_=130000,  # visualmap默认映射数据范围是【0，100】，需调整
            is_piecewise=True,
            range_color=["#CCD3D9", "#E6B6C2", "#D4587A", "#DC364C"],
        ))
        map_chart.render_notebook()
        map_chart.render("map_with_viusalmap.html")




if __name__ == "__main__" :
    map_with_viusalmap()
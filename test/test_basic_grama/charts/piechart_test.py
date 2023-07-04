from pyecharts import options as opts
from pyecharts.charts import Pie
from pyecharts.options import InitOpts, TitleOpts, LegendOpts, ToolboxOpts, LabelOpts

def basic_pie():
    # 构造数据
    data = [("A", 55), ("B", 20), ("C", 18), ("D", 5), ("E", 2)]

    # 使用链式写法创建Pie实例，添加数据并设置全局属性、系列属性
    pie = (Pie()
           .add("", data)
           .set_global_opts(title_opts=opts.TitleOpts(title="基础饼图"))
           .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
          )


    # 在jupyter notebook输出
    pie.render_notebook()

    # 在浏览器中显示图表
    pie.render("basic_pie.html")




def basic_pie_customercolor():
    # 构造数据
    data = [("A", 55), ("B", 20), ("C", 18), ("D", 5), ("E", 2)]

    # 使用链式写法创建Pie实例，添加数据并设置全局属性、系列属性
    pie = (Pie()
           .add("", data)
           .set_colors(["#c23531", "#2f4554", "#61a0a8", "#d48265", "#91c7ae"])  # 修改颜色
           .set_global_opts(title_opts=opts.TitleOpts(title="基础饼图"))
           .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
           )

    # 在jupyter notebook输出
    pie.render_notebook()

    # 在jupyter notebook输出
    pie.render_notebook()

    # 在浏览器中显示图表
    pie.render("basic_pie_customercolor.html")



def persent_pie_test():


    # 构造数据
    data = [("A", 55), ("B", 20), ("C", 18), ("D", 5), ("E", 2)]

    # 绘制饼图
    pie = (Pie()
           .add("", data)
           .set_global_opts(title_opts=opts.TitleOpts(title="百分比饼图"))
           .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {d}%"))
           )

    # 在jupyter notebook输出
    pie.render_notebook()
    pie.render("persent_pie_test.html")


def ring_pie_test():
    # 构造数据
    data = [("A", 55), ("B", 20), ("C", 18), ("D", 5), ("E", 2)]

    # 绘制饼图
    pie = (Pie()
           .add("", data, radius=["15%", "50%"])  # 设置饼图内圈和外圈的大小比例
           .set_global_opts(title_opts=opts.TitleOpts(title="环形饼图"))
           .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {d}%"))
           )

    # 在jupyter notebook输出
    pie.render_notebook()

    # 输出到jupyter notebook
    pie.render_notebook()
    pie.render("ring_pie_test.html")



def rose_pie_test():


    # 初始化Pie对象
    pie = Pie(init_opts=InitOpts(width='800px', height='400px'))

    # 添加数据
    data = [('类别1', 15), ('类别2', 20), ('类别3', 10), ('类别4', 5), ('类别5', 5), ('类别6', 5)]
    pie.add(series_name='', data_pair=data, radius=['30%', '70%'], rosetype='radius')

    # 设置全局配置项
    pie.set_global_opts(title_opts=TitleOpts(title='玫瑰饼图示例'),
                        legend_opts=LegendOpts(is_show=True),
                        toolbox_opts=ToolboxOpts(is_show=True),
                        )

    # 设置系列配置项
    pie.set_series_opts(label_opts=LabelOpts(formatter="{b}: {c} ({d}%)"))

    # 输出到jupyter notebook
    pie.render_notebook()
    pie.render("rose_pie_test.html")



def inner_ring_pie_test():


    # 定义内环数据
    inner_x_data = ["直达", "营销广告", "搜索引擎"]
    inner_y_data = [335, 679, 1548]
    inner_data_pair = [list(z) for z in zip(inner_x_data, inner_y_data)]

    # 定义外环数据
    outer_x_data = ["直达", "营销广告", "搜索引擎", "邮件营销", "联盟广告", "视频广告", "百度", "谷歌", "必应", "其他"]
    outer_y_data = [335, 310, 234, 135, 1048, 251, 147, 102]
    outer_data_pair = [list(z) for z in zip(outer_x_data, outer_y_data)]

    # 创建Pie对象
    pie = (
        Pie(init_opts=opts.InitOpts(width="800px", height="800px"))  # 设置图表大小
            .add(
            series_name="访问来源",  # 设置系列名称
            data_pair=inner_data_pair,  # 设置内环数据
            radius=[0, "30%"],  # 设置内外环半径
            label_opts=opts.LabelOpts(position="inner"),  # 设置标签位置
        )
            .add(
            series_name="访问来源",  # 设置系列名称
            radius=["40%", "55%"],  # 设置内外环半径
            data_pair=outer_data_pair,  # 设置外环数据
            label_opts=opts.LabelOpts(
                position="outside",  # 设置标签位置
                formatter="{a|{a}}{abg|}\n{hr|}\n {b|{b}: }{c}  {per|{d}%}  ",  # 设置标签格式
                background_color="#eee",  # 标签背景色
                border_color="#aaa",  # 标签边框颜色
                border_width=1,  # 标签边框宽度
                border_radius=4,  # 标签边框圆角半径
                rich={
                    "a": {"color": "#999", "lineHeight": 22, "align": "center"},
                    "abg": {
                        "backgroundColor": "#e3e3e3",
                        "width": "100%",
                        "align": "right",
                        "height": 22,
                        "borderRadius": [4, 4, 0, 0],
                    },
                    "hr": {
                        "borderColor": "#aaa",
                        "width": "100%",
                        "borderWidth": 0.5,
                        "height": 0,
                    },
                    "b": {"fontSize": 16, "lineHeight": 33},
                    "per": {
                        "color": "#eee",
                        "backgroundColor": "#334455",
                        "padding": [2, 4],
                        "borderRadius": 2,
                    },
                },
            ),
        )
            .set_global_opts(legend_opts=opts.LegendOpts(pos_left="left", orient="vertical"))  # 设置全局配置
            .set_series_opts(
            tooltip_opts=opts.TooltipOpts(
                trigger="item", formatter="{a} <br/>{b}: {c} ({d}%)"
            )  # 设置提示框格式
        )
    )

    # 输出到jupyter notebook
    pie.render_notebook()
    pie.render("inner_ring_pie_test.html")


def  mul_pies():


    c = (
        Pie()
            .add(
            "",
            [list(z) for z in zip(["剧情", "其他"], [30, 70])],
            center=["20%", "30%"],
            radius=[60, 80],
            label_opts=opts.LabelOpts(formatter="{b}: {d}%")  # 给每个饼图加上百分比
        )
            .add(
            "",
            [list(z) for z in zip(["奇幻", "其他"], [40, 60])],
            center=["55%", "30%"],
            radius=[60, 80],
            label_opts=opts.LabelOpts(formatter="{b}: {d}%")
        )
            .add(
            "",
            [list(z) for z in zip(["爱情", "其他"], [24, 76])],
            center=["20%", "70%"],
            radius=[60, 80],
            label_opts=opts.LabelOpts(formatter="{b}: {d}%")
        )
            .add(
            "",
            [list(z) for z in zip(["惊悚", "其他"], [11, 89])],
            center=["55%", "70%"],
            radius=[60, 80],
            label_opts=opts.LabelOpts(formatter="{b}: {d}%")
        )
            .set_global_opts(
            title_opts=opts.TitleOpts(title="Pie-多饼图基本示例"),
            legend_opts=opts.LegendOpts(
                type_="scroll", pos_top="20%", pos_left="80%", orient="vertical"
            ),
        )
    )

    c.render_notebook()
    c.render("mul_pies.html")


if __name__ == "__main__" :
    mul_pies()
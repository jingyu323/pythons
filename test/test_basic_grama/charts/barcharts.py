from pyecharts import options as opts  # 导入pyecharts的配置模块
from pyecharts.charts import Bar  # 导入柱状图模块
def roate_axi():


    # 定义x轴的数据
    x_data = ["语文", "数学", "英语", "历史", "政治", "地理", "体育"]
    # 定义学生A的成绩数据
    y_value1 = [99, 85, 90, 95, 70, 92, 98]
    # 定义学生B的成绩数据
    y_value2 = [90, 95, 88, 85, 96, 80, 76]

    # 创建柱状图实例
    bar = (
        Bar()  # 使用Bar()函数创建一个Bar对象
        .add_xaxis(x_data)  # 添加x轴数据
        .add_yaxis("学生A", y_value1)  # 添加学生A的成绩数据，设置系列名称为"学生A"
        .add_yaxis("学生B", y_value2)  # 添加学生B的成绩数据，设置系列名称为"学生B"
        .set_global_opts(title_opts=opts.TitleOpts(title="我是主标题"),
                         xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15))  # 设置x轴标签旋转角度为-15度
                         )  # 设置全局配置项，包括标题
    )

    # 在jupyter notebook中渲染图表
    bar.render_notebook()

if __name__ == "__main__" :
    roate_axi()
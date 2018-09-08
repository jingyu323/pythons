import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
创建一个简单的张量
一个张量（Tensor） 由两部分组成：
    1.dtype Tensor 存储的数据的类型，可以为tf.float32、tf.int32、tf.string
    2.shape Tensor 存储的多维数组中每个维度的数组中元素的个数，如上面例子中的shape
        
    3                                       # 这个 0 阶张量就是标量，shape=[]
    [1., 2., 3.]                            # 这个 1 阶张量就是向量，shape=[3]
    [[1., 2., 3.], [4., 5., 6.]]            # 这个 2 阶张量就是二维数组，shape=[2, 3]
    [[[1., 2., 3.]], [[7., 8., 9.]]]        # 这个 3 阶张量就是三维数组，shape=[2, 1, 3]

TensorFlow 中的数据流图有以下几个优点：
    1.可并行 计算节点之间有明确的线进行连接，系统比较容易判断哪些计算可以并行执行，
    2.可分发 途中不同的节点可以分布在不同的单元(CPU，GPU，TPU)或者不同的机器中，每个节点中产生的数据可以通过明确的线发个下一个节点中
    3.可优化  TensorFlow 中的 XLA 编译器可以根据数据流图进行代码优化，加快运行速度
    4.可移植 数据流图的信息可以不依赖代码进行保存，如使用Python创建的图，经过保存后可以在C++或Java中使用
    
Sesssion
    TensorFlow 底层是使用C++实现，这样可以保证计算效率，并使用 tf.Session类来连接客户端程序与C++运行时。
    上层的Python、Java等代码用来设计、定义模型，构建的Graph，最后通过tf.Session.run()方法传递给底层执行
    
    也就是说是一个客户端底层交互的一个桥梁
    
        

'''

# 创建两个常量节点
node1 = tf.constant(3.2)
node2 = tf.constant(4.8)
# 创建一个 adder 节点，对上面两个节点执行 + 操作
adder = node1 + node2
# 打印一下 adder 节点
print(adder)
# 打印 adder 运行后的结果
sess = tf.Session()
print(sess.run(adder))

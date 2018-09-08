import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
数据流图(Dataflow Graph)

是一种常见的并行计算编程模型，数据流图是由节点和线构成的有向图
    1.节点(nodes) 表示计算单元，也可以是输入的起点或者输出的终点
    2.线（edges）表示节点之间的输入输出关系
    
在 TensorFlow 中，每个节点都是用 tf.Tensor的实例来表示的，即每个节点的输入、输出都是Tensor


'''
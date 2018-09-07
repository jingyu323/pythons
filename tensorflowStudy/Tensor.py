import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
创建一个简单的张量
一个张量 由两部分组成：
    1.dtype Tensor 存储的数据的类型，可以为tf.float32、tf.int32、tf.string
    2.shape Tensor 存储的多维数组中每个维度的数组中元素的个数，如上面例子中的shape
        
    3                                       # 这个 0 阶张量就是标量，shape=[]
    [1., 2., 3.]                            # 这个 1 阶张量就是向量，shape=[3]
    [[1., 2., 3.], [4., 5., 6.]]            # 这个 2 阶张量就是二维数组，shape=[2, 3]
    [[[1., 2., 3.]], [[7., 8., 9.]]]        # 这个 3 阶张量就是三维数组，shape=[2, 1, 3]

'''

# 创建一个整型常量，即 0 阶 Tensor
t0 = tf.constant(3, dtype=tf.int32)

# 创建一个浮点数的一维数组，即 1 阶 Tensor
t1 = tf.constant([3., 4.1, 5.2], dtype=tf.float32)

# 创建一个字符串的2x2数组，即 2 阶 Tensor
t2 = tf.constant([['Apple', 'Orange'], ['Potato', 'Tomato']], dtype=tf.string)

# 创建一个 2x3x1 数组，即 3 阶张量，数据类型默认为整型
t3 = tf.constant([[[5], [6], [7]], [[4], [3], [2]]])

# 打印上面创建的几个 Tensor
print(t0)
print(t1)
print(t2)
print(t3)


#print 一个 Tensor 只能打印出它的属性定义，并不能打印出它的值，要想查看一个 Tensor 中的值还需要经过Session 运行一下：

sess = tf.Session()
print(sess.run(t0))
print(sess.run(t1))
print(sess.run(t2))
print(sess.run(t3))

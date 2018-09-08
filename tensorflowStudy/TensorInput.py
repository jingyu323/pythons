import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''

上面使用tf.constant()创建的 Tensor 都是常量，一旦创建后其中的值就不能改变了。
有时我们还会需要从外部输入数据，这时可以用tf.placeholder 创建占位 Tensor，
占位 Tensor 的值可以在运行的时候输入。如下就是创建占位 Tensor 的例子


'''
import tensorflow as tf
# 创建两个占位 Tensor 节点
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
# 创建一个 adder 节点，对上面两个节点执行 + 操作
adder = a + b
# 打印三个节点
print(a)
print(b)
print(adder)
# 运行一下，后面的 dict 参数是为占位 Tensor 提供输入数据
sess = tf.Session()
print(sess.run(adder, {a: 3, b: 4.5}))
print(sess.run(adder, {a: [1, 3], b: [2, 4]}))
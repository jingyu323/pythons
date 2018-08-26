import tensorflow as tf

OP_HELLO = tf.constant('Hello, Tensor Flow!')
# SESSION 的类型为 tensorflow.python.client.session.Session
SESSION = tf.Session()
# 输出结果为：
# b'Hello, Tensor Flow!'
print(SESSION.run(OP_HELLO))
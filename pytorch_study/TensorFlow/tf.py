import keras
import tensorflow

# 打印TensorFlow版本
print("TensorFlow version:", tensorflow.__version__)

# 打印Python版本
import sys

print("Python version:", sys.version)


import tensorflow as tf

print("TensorFlow Version:", tf.__version__)
print("Available devices: ", tf.config.experimental.list_physical_devices())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


print("Keras version:", keras.__version__)
gpus = tf.config.list_physical_devices("GPU")
print(gpus)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

print(tf.__version__)
print(tf.keras.__version__)

# 为了查出我们的操作和张量被配置到哪个 GPU 或 CPU 上，我们可以在程序起始位置加上：
tf.debugging.set_log_device_placement(True)

# 如果想要在所有 GPU 中指定只使用第一个 GPU，那么需要添加以下语句。

tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
# 设置 GPU 显存为固定使用量
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

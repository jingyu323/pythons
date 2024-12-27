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



gpus = tf.config.list_physical_devices("GPU")
print(gpus)
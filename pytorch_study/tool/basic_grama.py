import glob

from keras_preprocessing.image import ImageDataGenerator

# 读取目录
directory = '../Keras/'
for filename in glob.glob(directory + '/*'):
    print(filename)


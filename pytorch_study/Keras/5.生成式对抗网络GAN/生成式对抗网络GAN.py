# coding: utf-8

# ### GAN介绍
# 生成对抗网络（Generative Adversarial Networks，GAN）最早由 Ian Goodfellow 在 2014 年提出，是目前深度学习领域最具潜力的研究成果之一。它的核心思想是：同时训练两个相互协作、同时又相互竞争的深度神经网络（一个称为生成器 Generator，另一个称为判别器 Discriminator）来处理无监督学习的相关问题。  
# 
# 通常，我们会用下面这个例子来说明 GAN 的原理：将警察视为判别器，制造假币的犯罪分子视为生成器。一开始，犯罪分子会首先向警察展示一张假币。警察识别出该假币，并向犯罪分子反馈哪些地方是假的。接着，根据警察的反馈，犯罪分子改进工艺，制作一张更逼真的假币给警方检查。这时警方再反馈，犯罪分子再改进工艺。不断重复这一过程，直到警察识别不出真假，那么模型就训练成功了。  
# 
# GAN的变体非常多，我们就以深度卷积生成对抗网络（Deep Convolutional GAN，DCGAN）为例，自动生成 MNIST 手写体数字。
# 
# ### 判别器：
# 判别器的作用是判断一个模型生成的图像和真实图像比，有多逼真。它的基本结构就是如下图所示的卷积神经网络（Convolutional Neural Network，CNN）。对于 MNIST 数据集来说，模型输入是一个 28x28 像素的单通道图像。Sigmoid 函数的输出值在 0-1 之间，表示图像真实度的概率，其中 0 表示肯定是假的，1 表示肯定是真的。与典型的 CNN 结构相比，这里去掉了层之间的 max-pooling。这里每个 CNN 层都以 LeakyReLU 为激活函数。而且为了防止过拟合，层之间的 dropout 值均被设置在 0.4-0.7 之间，模型结构如下：
# <center><img src="images/Discriminator.jpg" alt="FAO" width="500"></center> 
# ReLU激活函数极为f(x)=alpha * x for x < 0, f(x) = x for x>=0。alpha是一个小的非零数。
# <center><img src="images/LeakyRelu.png" alt="FAO" width="200"></center>
# 
# ### 生成器：
# 生成器的作用是合成假的图像，其基本机构如下图所示。图中，我们使用了卷积的倒数，即[转置卷积（transposed convolution）](https://github.com/vdumoulin/conv_arithmetic)，从 100 维的噪声（满足 -1 至 1 之间的均匀分布）中生成了假图像。这里我们采用了模型前三层之间的上采样来合成更逼真的手写图像。在层与层之间，我们采用了批量归一化的方法来平稳化训练过程。以 ReLU 函数为每一层结构之后的激活函数。最后一层 Sigmoid 函数输出最后的假图像。第一层设置了 0.3-0.5 之间的 dropout 值来防止过拟合。
# <center><img src="images/Generator.jpg" alt="FAO" width="500"></center> 
# 批量正则化：
# <center><img src="images/batch normalization.png" alt="FAO" width="500"></center>
# 
# ### GAN应用
# [1.图像生成](http://make.girls.moe)  
# 2.向量空间运算
# <center><img src="images/GAN1.jpg" alt="FAO" width="500"></center>
# 3.文本转图像
# <center><img src="images/GAN2.jpg" alt="FAO" width="500"></center>
# 4.超分辨率
# <center><img src="images/GAN4.jpg" alt="FAO" width="500"></center>


import numpy as np
import matplotlib.pyplot as plt

from keras import Input, Model, Sequential
from keras.src.datasets import mnist
from keras.src.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense, Conv2D, LeakyReLU, Dropout, \
    Flatten, Activation, BatchNormalization, Reshape, UpSampling2D, Conv2DTranspose
from keras.src.optimizers import RMSprop
from keras.src.saving import load_model
from keras.src.utils import to_categorical


class DCGAN(object):
    def __init__(self, img_rows=28, img_cols=28, channel=1):
        # 初始化图片的行列通道数
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator 判别器
        self.G = None   # generator 生成器
        self.AM = None  # adversarial model 对抗模型
        self.DM = None  # discriminator model 判别模型

    # 判别模型
    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        # 定义通道数64
        depth = 64
        # dropout系数
        dropout = 0.4
        # 输入28*28*1
        input_shape = (self.img_rows, self.img_cols, self.channel)
        # 输出14*14*64
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))
        # 输出7*7*128
        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))
        # 输出4*4*256
        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))
        # 输出4*4*512
        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # 全连接层
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D

    # 生成模型
    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        # dropout系数
        dropout = 0.4
        # 通道数256
        depth = 64*4
        # 初始平面大小设置
        dim = 7
        # 全连接层，100个的随机噪声数据，7*7*256个神经网络
        self.G.add(Dense(dim*dim*depth, input_dim=100))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        # 把1维的向量变成3维数据(7,7,256)
        self.G.add(Reshape((dim, dim, depth)))
        self.G.add(Dropout(dropout))


        # 用法和 MaxPooling2D 基本相反，比如：UpSampling2D(size=(2, 2))
        # 就相当于将输入图片的长宽各拉伸一倍，整个图片被放大了
        # 上采样，采样后得到数据格式(14,14,256)
        self.G.add(UpSampling2D()) 
        # 转置卷积，得到数据格式(14,14,128) 
        self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same')) 
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # 上采样，采样后得到数据格式(28,28,128)
        self.G.add(UpSampling2D()) 
        # 转置卷积，得到数据格式(28,28,64) 
        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # 转置卷积，得到数据格式(28,28,32) 
        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same')) 
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # 转置卷积，得到数据格式(28,28,1) 
        self.G.add(Conv2DTranspose(1, 5, padding='same'))
        self.G.add(Activation('sigmoid'))
        self.G.summary()
        return self.G

    # 定义判别模型
    def discriminator_model(self):
        if self.DM:
            return self.DM
        # 定义优化器
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        # 构建模型
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.DM

    # 定义对抗模型
    def adversarial_model(self):
        if self.AM:
            return self.AM
        # 定义优化器
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        # 构建模型
        self.AM = Sequential()
        # 生成器
        self.AM.add(self.generator())
        # 判别器
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.AM

class MNIST_DCGAN(object):
    def __init__(self):
        # 图片的行数
        self.img_rows = 28
        # 图片的列数
        self.img_cols = 28
        # 图片的通道数
        self.channel = 1

        # 载入数据
        (x_train,y_train),(x_test,y_test) = mnist.load_data()
        # (60000,28,28)
        self.x_train = x_train/255.0
        # 改变数据格式(samples, rows, cols, channel)(60000,28,28,1)
        self.x_train = self.x_train.reshape(-1, self.img_rows, self.img_cols, 1).astype(np.float32)

        # 实例化DCGAN类
        self.DCGAN = DCGAN()
        # 定义判别器模型
        self.discriminator =  self.DCGAN.discriminator_model()
        # 定义对抗模型
        self.adversarial = self.DCGAN.adversarial_model()
        # 定义生成器
        self.generator = self.DCGAN.generator()

    # 训练模型
    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        noise_input = None
        if save_interval>0:
            # 生成16个100维的噪声数据
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        for i in range(train_steps):
        # 训练判别器，提升判别能力
            # 随机得到一个batch的图片数据
            images_train = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size), :, :, :]
            # 随机生成一个batch的噪声数据
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            # 生成伪造的图片数据
            images_fake = self.generator.predict(noise)
            # 合并一个batch的真实图片和一个batch的伪造图片
            x = np.concatenate((images_train, images_fake))
            # 定义标签，真实数据的标签为1，伪造数据的标签为0
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            # 把数据放到判别器中进行判断
            d_loss = self.discriminator.train_on_batch(x, y)
        
        # 训练对抗模型，提升生成器的造假能力
            # 标签都定义为1
            y = np.ones([batch_size, 1])
            # 生成一个batch的噪声数据
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            # 训练对抗模型
            a_loss = self.adversarial.train_on_batch(noise, y)
            # 打印判别器的loss和准确率，以及对抗模型的loss和准确率
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            # 如果需要保存图片
            if save_interval>0:
                # 每save_interval次保存一次
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0], noise=noise_input, step=(i+1))

    # 保存图片
    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'mnist.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "mnist_%d.png" % step
            # 生成伪造的图片数据
            images = self.generator.predict(noise)
        else:
            # 获得真实图片数据
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        # 设置图片大小
        plt.figure(figsize=(10,10))
        # 生成16张图片
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            # 获取一个张图片数据
            image = images[i, :, :, :]
            # 变成2维的图片
            image = np.reshape(image, [self.img_rows, self.img_cols])
            # 显示灰度图片
            plt.imshow(image, cmap='gray')
            # 不显示坐标轴
            plt.axis('off')
        # 保存图片
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        # 不保存的话就显示图片
        else:
            plt.show()

            
# 实例化网络的类
mnist_dcgan = MNIST_DCGAN()
# 训练模型
mnist_dcgan.train(train_steps=10000, batch_size=256, save_interval=500)



# mnist_dcgan.plot_images(fake=True)




mnist_dcgan.plot_images(fake=False)


mnist_dcgan.generator.save('generator.h5')
mnist_dcgan.discriminator.save('discriminator.h5')
mnist_dcgan.adversarial.save('adversarial.h5')






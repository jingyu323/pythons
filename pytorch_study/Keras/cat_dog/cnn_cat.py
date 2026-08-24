import os,shutil
from torchvision import transforms,datasets
import numpy as np
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
from torch.utils import data
# 数据图像生成
datagen = ImageDataGenerator(
    rotation_range=40,  # 随机旋转度数
    width_shift_range=0.2,  # 随机水平平移
    height_shift_range=0.2,  # 随机竖直平移
    rescale=1 / 255,  # 数据归一化
    shear_range=0.2,  # 随机裁剪
    zoom_range=0.2,  # 随机放大
    horizontal_flip=True,  # 水平翻转
    fill_mode='nearest',  # 填充方式
)

base_dir = 'E:/data/kreas/Kaggle/cat-dog-small/'
train_dir = os.path.join(base_dir, 'train/cats')

# 载入图片
image = load_img(train_dir+'/cat.1.jpg')
x = img_to_array(image)  # 图像数据是一维的，把它转成数组形式
print(x.shape)
x = np.expand_dims(x, 0)  # 在图片的0维增加一个维度，因为Keras处理图片时是4维,第一维代表图片数量
print(x.shape)

#生成20张图片数据
i=0
for batch in datagen.flow(x,batch_size=1,save_to_dir='temp',save_prefix='new_cat',save_format='jpeg'):
    i+=1
    if i==20:
        break
print('finshed!')


def mymovefile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("src not exist!")
    else:
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.move(srcfile, dstfile)  # 移动文件


test_rate = 0.2  # 训练集和测试集的比例为8:2。
img_num = 4000
test_num = int(img_num * test_rate)

import random

test_index = random.sample(range(0, img_num), test_num)
file_path = r"E:/data/kreas/Kaggle/cat-dog-small"
tr = "train"
te = "test"
cat = "cats"
dog = "dogs"

# 将上述index中的文件都移动到/test/Cat/和/test/Dog/下面去。
for i in range(len(test_index)):
    # 移动猫
    srcfile = os.path.join(file_path, tr, cat, "cat."+str(test_index[i]) + ".jpg")
    dstfile = os.path.join(file_path, te, cat, "cat."+str(test_index[i]) + ".jpg")

    mymovefile(srcfile, dstfile)
    # 移动狗
    srcfile = os.path.join(file_path, tr, dog, "dog."+str(test_index[i]) + ".jpg")
    dstfile = os.path.join(file_path, te, dog, "dog."+str(test_index[i]) + ".jpg")
    mymovefile(srcfile, dstfile)


# 定义transforms
transforms = transforms.Compose(
    [
transforms.RandomResizedCrop(150),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
]
)

train_data = datasets.ImageFolder(os.path.join(file_path, tr), transforms)
test_data = datasets.ImageFolder(os.path.join(file_path, te), transforms)


batch_size = 32
train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = data.DataLoader(test_data, batch_size=batch_size)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 架构会有很大的不同，因为28*28-》150*150,变化挺大的，这个步长应该快一点。
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 5)  # 和MNIST不一样的地方，channel要改成3，步长我这里加快了，不然层数太多。
        self.conv2 = nn.Conv2d(20, 50, 4, 1)
        self.fc1 = nn.Linear(50 * 6 * 6, 200)
        self.fc2 = nn.Linear(200, 2)  # 这个也不一样，因为是2分类问题。

    def forward(self, x):
        # x是一个batch_size的数据
        # x:3*150*150
        x = F.relu(self.conv1(x))
        # 20*30*30
        x = F.max_pool2d(x, 2, 2)
        # 20*15*15
        x = F.relu(self.conv2(x))
        # 50*12*12
        x = F.max_pool2d(x, 2, 2)
        # 50*6*6
        x = x.view(-1, 50 * 6 * 6)
        # 压扁成了行向量，(1,50*6*6)
        x = F.relu(self.fc1(x))
        # (1,200)
        x = self.fc2(x)
        # (1,2)
        return F.log_softmax(x, dim=1)

lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)


def train(model, device, train_loader, optimizer, epoch, losses):
    model.train()
    for idx, (t_data, t_target) in enumerate(train_loader):
        t_data, t_target = t_data.to(device), t_target.to(device)
        pred = model(t_data)  # batch_size*2
        loss = F.nll_loss(pred, t_target)

        # Adam
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 10 == 0:
            print("epoch:{},iteration:{},loss:{}".format(epoch, idx, loss.item()))
            losses.append(loss.item())

def test(model, device, test_loader):
    model.eval()
    correct = 0  # 预测对了几个。
    with torch.no_grad():
        for idx, (t_data, t_target) in enumerate(test_loader):
            t_data, t_target = t_data.to(device), t_target.to(device)
            pred = model(t_data)  # batch_size*2
            pred_class = pred.argmax(dim=1)  # batch_size*2->batch_size*1
            correct += pred_class.eq(t_target.view_as(pred_class)).sum().item()
    acc = correct / len(test_data)
    # print("accuracy:{},average_loss:{}".format(acc,average_loss))
    print("accuracy:{}".format(acc))

num_epochs = 10
losses = []
from time import *

begin_time = time()
for epoch in range(num_epochs):
    train(model, device, train_loader, optimizer, epoch, losses)

end_time = time()


print(test(model, device, test_loader))
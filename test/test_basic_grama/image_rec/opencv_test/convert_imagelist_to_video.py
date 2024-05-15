import os

import cv2
import numpy as np


def resize_img():
    target_size = (900, 800)  # 所有图片缩放设置一致尺寸，目标尺寸
    path = './image'
    path_new = './images_new'
    if not os.path.exists(path_new):
        os.makedirs(path_new)
    filelists = []
    imglist = []
    for i in os.listdir(path):
        file_path = os.path.join(path, i)
        print(file_path)
        img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        size = img.shape
        h, w = size[0], size[1]
        target_h, target_w = target_size[1], target_size[0]
        # 确定缩放的尺寸
        scale_h, scale_w = float(h / target_h), float(w / target_w)
        print(f'scale_h:{scale_h}, scale_w:{scale_w}')
        scale = max(scale_h, scale_w)  # 选择最大的缩放比率
        new_w, new_h = int(w / scale), int(h / scale)
        # 确定缩放的尺寸
        scale_h, scale_w = float(h / target_h), float(w / target_w)
        print(f'scale_h:{scale_h}, scale_w:{scale_w}')
        scale = max(scale_h, scale_w)  # 选择最大的缩放比率
        new_w, new_h = int(w / scale), int(h / scale)
        # 缩放后其中一条边和目标尺寸一致
        resize_img = cv2.resize(img, (new_w, new_h))
        # 图像上、下、左、右边界分别需要扩充的像素数目
        top = int((target_h - new_h) / 2)
        bottom = target_h - new_h - top
        left = int((target_w - new_w) / 2)
        right = target_w - new_w - left
        print(f'top:{top} bottom:{bottom} left:{left} right:{right}')
        cv2.imwrite(os.path.join(path_new, f'new_{i}'), resize_img)  # 写入本地文件
        # 填充至 target_w * target_h
        pad_img = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # cv2.imshow('img', pad_img)
        # cv2.waitKey(1000)
        filelists.append(os.path.join(path_new, f'new_{i}'))
        imglist.append(pad_img)
    return filelists, imglist


def cut_img(scale):
    path = './image'
    path_new = './images_new'
    if not os.path.exists(path_new):
        os.makedirs(path_new)
    filelists = []
    imglist = []
    for i in os.listdir(path):
        file_path = os.path.join(path, i)
        print(file_path)
        img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        size = img.shape
        h, w = size[0], size[1]
        rate1 = scale.split(':')
        w1 = int((w - w * int(rate1[0]) / int(rate1[1])) / 2)
        w2 = int(w - (w - w * int(rate1[0]) / int(rate1[1])) / 2)
        resize_img = img[0:h, w1:w2]
        cv2.imwrite(os.path.join(path_new, f'new_{i}'), resize_img)  # 写入本地文件
        filelists.append(os.path.join(path_new, f'new_{i}'))
        imglist.append(resize_img)
    return filelists, imglist

def image_to_video():
    scale = '1:1'  # 裁剪比例,并保持高度不变
    # scale = '3:4'
    # scale = '9:16'
    # filelists, imglist = cut_img(scale)  # 裁剪

    filelists, imglist = resize_img() # 缩放
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    im = cv2.imread(filelists[0])
    print(im.shape)
    shape1 = (im.shape[1], im.shape[0])  # 需要转为视频的图片的尺寸, 视频的分辨率
    print('shape1:', shape1)
    fps = 1
    writer = cv2.VideoWriter('./output.mp4', fourcc, fps, shape1)
    # for file_path in filelists:
    #     img = cv2.imread(file_path)
    #     writer.write(img)

    for i in imglist:
        writer.write(i)
    writer.release()

if __name__ == '__main__':
     image_to_video()
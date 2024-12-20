import matplotlib.pyplot as plt

from PIL import Image

import os
import glob
import numpy as np
from keras.src.legacy.preprocessing.image import ImageDataGenerator


def print_result(path):
    name_list = glob.glob(path)
    fig = plt.figure(figsize=(120,160))
    for i in range(3):
        img = Image.open(name_list[i])
        sub_img = fig.add_subplot(131+i)
        sub_img.imshow(img)
    print(fig)
img_path = './image/*'
in_path = './image/'
out_path = './output/'
name_list = glob.glob(img_path)
print("name_list=",name_list)
print_result(img_path)


datagen = ImageDataGenerator()


gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path + 'rotation_range',
                                       save_prefix='gen', target_size=(224, 224))

for i, gen_datum in enumerate(gen_data):
    print(gen_datum)


for i in range(3):
    gen_data.__next__()
    print_result(out_path+'resize/*')


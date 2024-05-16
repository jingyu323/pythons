import cv2
import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"


def load_dataset(image_dir):
    '''
    This function loads in images and their labels and places them in a list
    image_dir:directions where images stored
    '''
    im_list = []
    image_types = ['red', 'yellow', 'green']

    # Iterate through each color folder
    for im_type in image_types:
        file_lists = glob.glob(os.path.join(image_dir, im_type, '*'))
        print(len(file_lists))
        for file in file_lists:
            im = mpimg.imread(file)

            if not im is None:
                im_list.append((im, im_type))
    return im_list


IMAGE_LIST = load_dataset(IMAGE_DIR_TRAINING)

_,ax = plt.subplots(1,3,figsize=(5,2))
#red
img_red = IMAGE_LIST[0][0]
ax[0].imshow(img_red)
ax[0].annotate(IMAGE_LIST[0][1],xy=(2,5),color='blue',fontsize='10')
ax[0].axis('off')
ax[0].set_title(img_red.shape,fontsize=10)
#yellow
img_yellow = IMAGE_LIST[730][0]
ax[1].imshow(img_yellow)
ax[1].annotate(IMAGE_LIST[730][1],xy=(2,5),color='blue',fontsize='10')
ax[1].axis('off')
ax[1].set_title(img_yellow.shape,fontsize=10)
#green
img_green = IMAGE_LIST[800][0]
ax[2].imshow(img_green)
ax[2].annotate(IMAGE_LIST[800][1],xy=(2,5),color='blue',fontsize='10')
ax[2].axis('off')
ax[2].set_title(img_green.shape,fontsize=10)
plt.show()


def standardize(image_list):
    '''
    This function takes a rgb image as input and return a standardized version
    image_list: image and label
    '''
    standard_list = []
    #Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]
        # Standardize the input
        standardized_im = standardize_input(image)
        # Standardize the output(one hot)
        one_hot_label = one_hot_encode(label)
        # Append the image , and it's one hot encoded label to the full ,processed list of image data
        standard_list.append((standardized_im,one_hot_label))
    return standard_list

def standardize_input(image):
    #Resize all images to be 32x32x3
    standard_im = cv2.resize(image,(32,32))
    return standard_im
def one_hot_encode(label):
    #return the correct encoded label.
    '''
    # one_hot_encode("red") should return: [1, 0, 0]
    # one_hot_encode("yellow") should return: [0, 1, 0]
    # one_hot_encode("green") should return: [0, 0, 1]
    '''
    if label=='red':
        return [1,0,0]
    elif label=='yellow':
        return [0,1,0]
    else:
        return [0,0,1]


def create_feature(rgb_image):
    '''
    Basic brightness feature
    rgb_image : a rgb_image
    '''
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    sum_brightness = np.sum(hsv[:, :, 2])
    area = 32 * 32
    avg_brightness = sum_brightness / area  # Find the average
    return avg_brightness


#Visualize
image_num = 0
test_im = Standardized_Train_List[image_num][0]
test_label = Standardized_Train_List[image_num][1]
#convert to hsv
hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)
# Print image label
print('Label [red, yellow, green]: ' + str(test_label))
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]
# Plot the original image and the three channels
_, ax = plt.subplots(1, 4, figsize=(20,10))
ax[0].set_title('Standardized image')
ax[0].imshow(test_im)
ax[1].set_title('H channel')
ax[1].imshow(h, cmap='gray')
ax[2].set_title('S channel')
ax[2].imshow(s, cmap='gray')
ax[3].set_title('V channel')
ax[3].imshow(v, cmap='gray')




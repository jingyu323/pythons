import cv2
import numpy as np


def np_copy_test():
    print()

    img = cv2.imread('../image/cat.jpg')
    # 浅拷贝
    img2 =  img.view()
    # 深拷贝
    img3 =  img.copy()

    img[10:100,10:100] = [0,0,255]
    cv2.imshow("img",img)
    cv2.imshow("img2", img2)
    cv2.imshow("img3", img3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
def img_split_merge():
    img = np.zeros((200,200,3),np.uint8)
    b,g,r= cv2.split(img)
    b[10:100,10:100] = 255
    g[10:100, 10:100] = 255

    img2= cv2.merge((b,g,r))

    cv2.imshow("img1", np.hstack((img,img2)))
    # cv2.imshow("img2",img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





if __name__ == '__main__':
    img_split_merge()
import cv2
import numpy as np


def masaike():
    cat = cv2.imread('../image/new_cat.jpg')
    img = np.repeat(cat,10,axis=0)
    img2 = np.repeat(img,10,axis=1)
    cv2.imshow("cat",img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def face_masike():
    cat = cv2.imread('../image/bao.jpeg')
    print(cat.shape)
    #  对于一个图片的二维数组，一个是高宽
    # 所以对一个图片的截取 也是先从高度中取出一部分
    face = cat[5:260,200:383]
    # cv2.imshow("cat", face)
    face = face[::10,::10]
    face = np.repeat(face,10,axis=0)
    face2 = np.repeat(face,10,axis=1)
    face2=face2[:255,:183]
    cat[5:260, 200:383]= face2

    cv2.imshow("cat",cat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    face_masike()
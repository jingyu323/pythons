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

def drw_shape():
    blank = np.zeros((224, 224, 3), np.uint8)

    # draw circle
    cv2.circle(blank, (112, 112), 12, (255, 255, 255), 1)
    cv2.imwrite("circle.png", blank)

    blank = np.zeros((224, 224, 3), np.uint8)

    # draw rectangle
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blank, "helloworld", (80, 90), font, 0.5, (255, 255, 255), 1)
    cv2.imshow("text.png", blank)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def drw_img():
    cat = cv2.imread('../image/cat.jpg')

    logo = np.zeros((200, 200, 3), np.uint8)
    logo[20:120,20:120] =[0,0,255]
    logo[80:180, 80:180] = [0, 255,0]

    mask = np.zeros((200, 200), np.uint8)
    mask[20:120,20:120] =255
    mask[80:180, 80:180] = 255
    cv2.imshow("mask", mask)
    # 取原图的一部分
    roi = cat[0:200,0:200]
    m= cv2.bitwise_not(mask)
    cv2.imshow("m", m)

    tmp = cv2.bitwise_and(roi,roi,mask=m)

    cv2.imshow("logo", logo)

    cv2.imshow("tmp", tmp)

    dst = cv2.add(tmp,logo)

    cv2.imshow("dst", dst)
    cat[:200,:200] =dst
    cv2.imshow("cat", cat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def drw_img2():
    cat = cv2.imread('../image/cat.jpg')

    logo = np.zeros((200, 200, 3), np.uint8)
    logo[20:120,20:120] =[0,0,255]
    logo[80:180, 80:180] = [0, 255,0]

    mask = np.zeros((200, 200), np.uint8)
    mask[20:120,20:120] =255
    mask[80:180, 80:180] = 255
    roi = cat[0:200,0:200]
    m= cv2.bitwise_not(mask)


    tmp = cv2.bitwise_and(roi,roi,mask=m)


    dst = cv2.add(tmp,logo)

    cat[:200,:200] =logo
    cv2.imshow("cat", cat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def matrix():
    cat = cv2.imread('../image/cat.jpg')

    h,w,ch = cat.shape

    M = np.float32([[1,0,200],[0,1,10]])

    new = cv2.warpAffine(cat,M,dsize=(w,h))
    # cv2.imshow("11cat", cat)
    cv2.imshow("new", new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#  透视变换 实现照片扶正
def warp_trans():
    cat = cv2.imread('../image/new_cat.jpg')

    # 浮雕
    # kenel = np.array([[-2,1,0],[-1,1,1],[0,1,2]])
    # 锐化
    kenel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])

    new = cv2.filter2D(cat,-1,kenel)

    cv2.imshow("new", np.hstack((cat,new)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#  去掉噪点，但是会模糊图片
def gus_filter():
    cat = cv2.imread('../image/new_cat.jpg')

    new = cv2.GaussianBlur(cat,(5,5),sigmaX=10)

    cv2.imshow("new", np.hstack((cat,new)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# 中值滤波
def medianBlur_filter():
    cat = cv2.imread('../image/paper.png')

    new = cv2.medianBlur(cat,5)
# 高斯只是进行模糊 但是去噪点效果不行
    new1 = cv2.GaussianBlur(cat, (5, 5), sigmaX=10)
    cv2.imshow("new", np.hstack((cat,new,new1)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#双边滤波 对降噪没什么效果
def bilateral_filter():
    cat = cv2.imread('../image/paper.png')

    new = cv2.bilateralFilter(cat,7,20,50)
# 高斯只是进行模糊 但是去噪点效果不行
#     new1 = cv2.GaussianBlur(cat, (5, 5), sigmaX=10)
    new2 = cv2.bilateralFilter(cat, 10, 40, 90)
    cv2.imshow("new", np.hstack(( new,cat,new2)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#双边滤波 对降噪没什么效果
def solbe():
    cat = cv2.imread('../image/qipan.png')
    # cv2.imshow("cat", cat)
    dx = cv2.Sobel(cat,cv2.CV_64F,dx=1,dy=0,ksize=3)
    dy = cv2.Sobel(cat,cv2.CV_64F,dx=0,dy=1,ksize=3)
# 高斯只是进行模糊 但是去噪点效果不行
#     ds = cv2.add(dy,dx)
    ds = cv2.addWeighted(dx,0.5,dy,0.5,gamma=0)
#
    cv2.imshow("cat", cat)
    cv2.imshow("new", dx)
    cv2.imshow("dy", dy)
    cv2.imshow("ds", ds)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 只能是3X3 的kenel 比较擅长比较细的边缘
def scharr():
    cat = cv2.imread('../image/paper.png')
    # cv2.imshow("cat", cat)
    dx = cv2.Scharr(cat,cv2.CV_64F,dx=1,dy=0 )
    dy = cv2.Scharr(cat,cv2.CV_64F,dx=0,dy=1 )
# 高斯只是进行模糊 但是去噪点效果不行
#     ds = cv2.add(dy,dx)
    ds = cv2.addWeighted(dx,0.5,dy,0.5,gamma=0)
#
    cv2.imshow("cat", cat)
    cv2.imshow("new", dx)
    cv2.imshow("dy", dy)
    cv2.imshow("ds", ds)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 只能是3X3 的kenel 比较擅长比较细的边缘
    def scharr():
        cat = cv2.imread('../image/paper.png')
        # cv2.imshow("cat", cat)
        dx = cv2.Scharr(cat, cv2.CV_64F, dx=1, dy=0)
        dy = cv2.Scharr(cat, cv2.CV_64F, dx=0, dy=1)
        # 高斯只是进行模糊 但是去噪点效果不行
        #     ds = cv2.add(dy,dx)
        ds = cv2.addWeighted(dx, 0.5, dy, 0.5, gamma=0)
        #
        cv2.imshow("cat", cat)
        cv2.imshow("new", dx)
        cv2.imshow("dy", dy)
        cv2.imshow("ds", ds)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    scharr()

## 图像轮廓
import cv2
import numpy as np
import  matplotlib.pyplot as plt


def count_find():
    cat = cv2.imread('../image/lunkuo.png')

    gray = cv2.cvtColor(cat,cv2.COLOR_BGR2GRAY)

    #  返回两个值 一个是阈值 一个是 二值化之后的值

    thresh,binary = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)

    #  层级  hier
    cou, hier = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    print(cou)

    copy = cat.copy()
    # -1   是所有  0 ， 1 是从外网内画轮廓
    cv2.drawContours(copy,cou,1, (0,0,255),2)

    area = cv2.contourArea(cou[1])

    print(area)

    perimeter = cv2.arcLength(cou[1],closed=True)

    print(perimeter)

    cv2.imshow("cat",  cat)
    cv2.imshow("copy",  copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def tu_bao():
    cat = cv2.imread('../image/hand.png')

    gray = cv2.cvtColor(cat,cv2.COLOR_BGR2GRAY)

    #  返回两个值 一个是阈值 一个是 二值化之后的值

    thresh,binary = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)

    #  层级  hier
    cou, hier = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # -1   是所有  0 ， 1 是从外网内画轮廓
    cv2.drawContours(cat,cou,0, (0,0,255),2)

    # 多边形逼近
    approx = cv2.approxPolyDP(cou[0],20,True)
    # -1   是所有  0 ， 1 是从外网内画轮廓
    cv2.drawContours(cat,[approx],0, (0,255,0),2)
    # 计算凸包
    hull = cv2.convexHull(cou[0])
    cv2.drawContours(cat, [hull], 0, (255, 0, 0), 2)
    cv2.imshow("cat",  cat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




# 最大外接矩形
def max_rec():
    cat = cv2.imread('../image/hello.png')

    gray = cv2.cvtColor(cat,cv2.COLOR_BGR2GRAY)

    #  返回两个值 一个是阈值 一个是 二值化之后的值

    thresh,binary = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)

    #  层级  hier
    cou, hier = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # -1   是所有  0 ， 1 是从外网内画轮廓
    # cv2.drawContours(cat,cou,1, (0,0,255),2)
    rec = cv2.minAreaRect(cou[1])

    ## 计算旋转坐标
    box  = cv2.boxPoints(rec)
    box = np.round(box).astype('int64')
    # 最小外接矩形
    cv2.drawContours(cat, [box], 0, (255, 0, 0), 2)
    # 最大外接矩形
    x,y,w,h = cv2.boundingRect(cou[1])
    cv2.rectangle(cat,(x,y),(x+w,y+h),(0, 255, 0),2)
    cv2.imshow("cat",  cat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# 金字塔
def pyramid():
    cat = cv2.imread('../image/paper.png')
    dst= cv2.pyrDown(cat)
    cv2.imshow("dst",  dst)
    cv2.imshow("cat",  cat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def pyramid_up():
    cat = cv2.imread('../image/paper.png')
    dst= cv2.pyrUp(cat)
    cv2.imshow("dst",  dst)
    cv2.imshow("cat",  cat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#  拉普拉斯金字塔用于图片压缩


#  直方图
def hits():
    cat = cv2.imread('../image/paper.png')

    gray = cv2.cvtColor(cat,cv2.COLOR_BGR2GRAY)

    print(gray)
    plt.hist(gray.ravel(),bins=256,range=[0,255])
    plt.show()
    # hits = cv2.calcHist([cat],[0],None,[256],[0,255])

    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

#  直方图
def calsc_ts():
    cat = cv2.imread('../image/paper.png')



    hitb = cv2.calcHist([cat],[0],None,[256],[0,255])
    hitg = cv2.calcHist([cat],[1],None,[256],[0,255])
    hitr = cv2.calcHist([cat],[2],None,[256],[0,255])

    plt.plot(hitb,color='b',label='blue')
    plt.plot(hitg,color='g',label='green')
    plt.plot(hitr,color='r',label='red')


    plt.show()
if __name__ == '__main__':
    calsc_ts()
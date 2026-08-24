import cv2
import numpy as np


def shif_test():

    img = cv2.imread('../image/qipan.png')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sifi = cv2.SIFT_create()

    kp = sifi.detect(gray)

    cv2.drawKeypoints(gray,kp,img)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def shif_tomas():

    img = cv2.imread('../image/qipan.png')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray,maxCorners=0   ,qualityLevel=0.1,minDistance=10)

    corners = np.int64(corners)

    for i in corners:
        x,y = i.ravel()
        cv2.circle(img,(x,y),3,(0,0,255),-1)


    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    shif_tomas()
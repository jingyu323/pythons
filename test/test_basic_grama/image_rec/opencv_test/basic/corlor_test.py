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

def auto_face_masike():
    cat = cv2.imread('../image/hezhao.jpeg')
    print(cat.shape)
    #  对于一个图片的二维数组，一个是高宽
    # 所以对一个图片的截取 也是先从高度中取出一部分

    #  人脸特征详细说明
    face_detector= cv2.CascadeClassifier("../xml/haarcascade_frontalface_alt.xml")

    faces = face_detector.detectMultiScale(cat,scaleFactor=1.14,minNeighbors=3)

    print(faces)

    star = cv2.imread('../image/star.jpeg')

    for (x,y,w,h) in faces:
        cv2.rectangle(cat,(x,y),(x+w,y+h),(0,255,0),2)

        cat[y:y+h//4,x+(3*w//8):x+w//4+(3*w//8)] = cv2.resize(star,(w//4,h//4))


    cv2.imshow("cat",cat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def auto_face_removebg():
    cat = cv2.imread('../image/hezhao.jpeg')
    print(cat.shape)
    #  对于一个图片的二维数组，一个是高宽
    # 所以对一个图片的截取 也是先从高度中取出一部分

    #  人脸特征详细说明
    face_detector = cv2.CascadeClassifier("../xml/haarcascade_frontalface_alt.xml")

    faces = face_detector.detectMultiScale(cat, scaleFactor=1.14, minNeighbors=3)

    print(faces)

    star = cv2.imread('../image/star.jpeg')

    for (x, y, w, h) in faces:
        cv2.rectangle(cat, (x, y), (x + w, y + h), (0, 255, 0), 2)

        re_star= cv2.resize(star, (w // 4, h // 4))
        w1= w // 4
        h1= h // 4
        for i in range(w1):
            for j in range(h1):
                if not (re_star[i][j] > 180).all() :
                    cat[ y +i,  j+x  + (3 * w // 8)]=re_star[i][j]

        # cat[y:y + h // 4, x + (3 * w // 8):x + w // 4 + (3 * w // 8)] = cv2.resize(star, (w // 4, h // 4))

    cv2.imshow("cat", cat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def dog_replace():
    cat = cv2.imread('../image/han.jpg')
    print(cat.shape)
    #  对于一个图片的二维数组，一个是高宽
    # 所以对一个图片的截取 也是先从高度中取出一部分

    #  人脸特征详细说明
    face_detector = cv2.CascadeClassifier("../xml/haarcascade_frontalface_alt.xml")
    gray = cv2.cvtColor(cat,cv2.COLOR_BGR2GRAY)

    thre,binary =  cv2.threshold(gray,150,255,cv2.THRESH_OTSU)

    faces = face_detector.detectMultiScale(gray, scaleFactor=1.14, minNeighbors=3)
    dog = cv2.imread('../image/goutou.png')
    dog_gray = cv2.cvtColor(dog, cv2.COLOR_BGR2GRAY)
    dog_thre,dog_binary =  cv2.threshold(dog_gray,150,255,cv2.THRESH_OTSU)

    dog_couters,dog_her =cv2.findContours(dog_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas=[]
    for cou in dog_couters:
        areas.append(cv2.contourArea(cou))

    areas = np.array(areas)
    index = areas.argsort()
    mask = np.zeros_like(dog_binary,dtype=np.uint8)
    # 最后移问-1 会将整个轮廓内部填充为白色
    mask=cv2.drawContours(mask,dog_couters,index[-2],(255,255,255),-1)
    cv2.imshow("mask", mask)
    for (x,y,w,h) in faces:
        mask2 = cv2.resize(mask,(w,h))
        dog_gray2 = cv2.resize(dog_gray,(w,h))
        dog2 = cv2.resize(dog,(w,h))
        for  i in range(h):
            for j in range(w):
                if((mask2[i,j])==255).all():
                    cat[i+y,j+x]=dog2[i,j]


    cv2.imshow("cat", cat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    dog_replace()
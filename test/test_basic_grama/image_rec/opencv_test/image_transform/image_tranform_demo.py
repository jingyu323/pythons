import cv2
import numpy as np


def mouse_event(event, x, y, flags, param ):
    img=param[0]
    img_hsv=param[1]
    if event == cv2.EVENT_LBUTTONDBLCLK:  # 双击左键显示图像的坐标和对应的rgb值
        print('img pixel value at(', x, ',', y, '):', img[y, x])  # 坐标(x,y)对应的像素值应该是img[y,x]
        text = '(' + str(x) + ',' + str(y) + ')' + str(img[y, x])
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # 绘制文字

    if event == cv2.EVENT_RBUTTONDBLCLK:  # 双击右键显示图像的坐标和对应的hsv值
        print('img_hsv pixel value at(', x, ',', y, '):', img_hsv[y, x])  # 坐标(x,y)对应的像素值应该是img_hsv[y,x]
        text = '(' + str(x) + ',' + str(y) + ')' + str(img_hsv[y, x])
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # 绘制文字

def  house_img_check():
     img = cv2.imread('../image/house.png')
     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  #转换到hsv空间

     cv2.namedWindow('image', cv2.WINDOW_NORMAL)  # 定义窗口
     # cv2.resizeWindow('image',(800,800))

     cv2.setMouseCallback('image', mouse_event,[img,img_hsv])  # 鼠标回调

     while True:
         cv2.imshow('image', img)
         if cv2.waitKey(1) == ord('q'):
             break
     cv2.destroyAllWindows()


def book_trans_form():
    img_book = cv2.imread('../image/bookf.png')
    h, w, c = img_book.shape

    src = np.float32([[308, 207], [62, 424], [304, 617], [522, 319]])  # 用微信截图来确定roi区域的四个角的坐标点位置(或者用鼠标事件输出四角坐标)
    # dst = np.float32([[300, 200], [300, 500], [510, 500], [510, 200]])
    dst = np.float32([[0,0],[0,300],[210,300],[210,0]])  #只保留roi区域，warpPerspective函数中尺寸设置为roi的宽高

    M = cv2.getPerspectiveTransform(src, dst)  # 透视变换矩阵
    print(M)

    # new_book = cv2.warpPerspective(img_book, M, (w, h))  # 透视变换  二维平面获得接近三维物体的视觉效果的算法
    new_book = cv2.warpPerspective(img_book,M,(210,300)) #透视变换  变换后的roi区域的宽高，只保留roi区域

    cv2.imshow('img_book', img_book)
    cv2.imshow('new_book', new_book)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#  根据最终图片大小 计算变换矩阵，变换完成之后 将嵌入区域替换成黑色 再执行图片相加
def change_ad_img():
    img2 = cv2.imread('../image/house.png')
    img1 = cv2.imread('../image/guagnb.jpg')
    h1, w1, c1 = img1.shape

    src = np.float32([[0, 0], [0, h1 - 1], [w1 - 1, h1 - 1], [w1 - 1, 0]])  # 需要嵌入的原图片的四角坐标（img1） 逆时针坐标
    dst = np.float32([[438, 53], [436, 397], [817, 528], [808, 378]])  # 待嵌入图片区域的位置坐标（img2）
    M = cv2.getPerspectiveTransform(src, dst)  # 透视变换矩阵

    h2, w2, c2 = img2.shape
    new_img1 = cv2.warpPerspective(img1, M, (w2, h2))  # 透视变换  二维平面获得接近三维物体的视觉效果的算法
    dst = dst.astype(int)  # 多边形的坐标需要整型
    cv2.imshow('new_img1', new_img1)
    cv2.fillConvexPoly(img2, dst, [0, 0, 0])  # 用多边形填充的办法把嵌入区域的像素全部变成0
    new_img = cv2.add(img2, new_img1)  # 把变换后的图片插入到广告牌的位置
    cv2.imshow('img1', img1)

    cv2.imshow('new_img', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#      试一下像素替换   因为图片不规则不好进行替换 所以放弃


def  image_search():
    img1 = cv2.imread('../image/cg.jpg')
    img2 = cv2.imread('../image/dgg.png')

    img_gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#    创建特征检测

    sift = cv2.SIFT_create()
    #  计算特征和描述算子
    kp1,des1 = sift.detectAndCompute(img_gray1, None)
    kp2,des2 = sift.detectAndCompute(img_gray2, None)
    # 创建特征匹配器
    index_param = dict(algorithm=1,trees=5)
    searche_param= dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_param,searche_param)
    # 对描述子进行特征匹配
    matches = flann.knnMatch(des1,des2,k=2)
    goods = []  # 选择两个匹配对象中好一些的保存下来

    for m,n in matches:
        if m.distance < n.distance * 0.75:
            goods.append(m)

    # 把找到的匹配特征点保存在goods中，注意单应性矩阵要求最少4个点
    if len(goods) >= 4:

        src_points=np.float32([kp1[m.queryIdx].pt  for m in goods]).reshape(-1,1,2)
        des_points=np.float32([kp2[m.trainIdx].pt for m in goods]).reshape(-1,1,2)
        print('des_points:',des_points)

        # 根据匹配上的关键点去计算单应性矩阵
        H, mask= cv2.findHomography(src_points,des_points,cv2.RANSAC,5)

        h, w = img1.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)  # img1的四个边框顶点，逆时针（坐标从0开始，因此要w-1,h-1）
        # 用透视变换函数找到这四个顶点对应的img2的位置，不用warpPerspective,这是对图像的透视变换，用perspectiveTransform()
        dst = cv2.perspectiveTransform(pts, H)
        print('pts:', pts)
        print('dst:', dst)
        # 在大图中把dst画出来，polylines
        cv2.polylines(img2, [np.int32(dst)], True, [0, 0, 255], 2)
    else:
        print('matches is not enough')
        exit()

    res = cv2.drawMatchesKnn(img1, kp1, img2, kp2, [goods], None)  # 画出匹配的特征点

    cv2.imshow('res', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def img_cpncat():
    img1 = cv2.imread('../image/hilll.png')
    img2 = cv2.imread('../image/hillr.png')

    img_gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 创建特征检测器
    sift = cv2.SIFT_create()

    # 计算特征点和描述子
    kp1,des1= sift.detectAndCompute(img_gray1,None)
    kp2,des2= sift.detectAndCompute(img_gray2,None)
    # 创建特征匹配器
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 对描述子进行特征匹配
    matches = flann.knnMatch(des1, des2, k=2)  # 用的knnmatch匹配

    goods = []  # 选择两个匹配对象中好一些的保存下来
    for (m, n) in matches:
        if m.distance < n.distance * 0.75:
            goods.append(m)

    print('goods', len(goods))


    # 把找到的匹配特征点保存在goods中，注意单应性矩阵要求最少4个点
    if len(goods) >= 4:
        # 把goods中的第一幅图和第二幅图的特征点坐标拿出来（坐标要float32且是三维矩阵类型  reshape(-1,1,2)） goods是Dmatch对象
        src_points = np.float32([kp1[m.queryIdx].pt for m in goods]).reshape(-1, 1, 2)
        des_points = np.float32([kp2[m.trainIdx].pt for m in goods]).reshape(-1, 1, 2)
        # print('des_points:',des_points)

        # 根据匹配上的关键点去计算单应性矩阵  第一个图对变成第二个图的视角计算出来的单应性矩阵
        H, mask = cv2.findHomography(src_points, des_points, cv2.RANSAC, 5)  # 参数5表示：允许有5个关键点的误差

    #     #利用H矩阵的逆求解视角和img1特征匹配的点的img2图，并且img1没有像素
    #     result = cv2.warpPerspective(img2,np.linalg.inv(H),(img1.shape[1]+img2.shape[1],img1.shape[0]+img2.shape[0]))
    #     direct = result.copy()
    #     direct[0:img1.shape[0],0:img1.shape[1]] = img1 #将左边的img1的部分重新赋值
    else:
        exit()

    # 获取原始图像的高宽
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    # 获取两幅图的边界坐标
    img1_pts = np.float32([[0, 0], [0, h1 - 1], [w1 - 1, h1 - 1], [w1 - 1, 0]]).reshape(-1, 1, 2)
    img2_pts = np.float32([[0, 0], [0, h2 - 1], [w2 - 1, h2 - 1], [w2 - 1, 0]]).reshape(-1, 1, 2)

    # 获取img1的边界坐标变换之后的坐标
    img1_transform = cv2.perspectiveTransform(img1_pts, H)
    # print('img1_pts',img1_pts)
    # print('img1_transform',img1_transform)
    # 把img2和转换后的边界坐标连接起来
    result_pts = np.concatenate((img2_pts, img1_transform), axis=0)
    print(result_pts)
    # result_pts.min(axis=0)  #对行聚合，返回每一列的最小值
    [x_min, y_min] = np.int32(result_pts.min(axis=0).ravel() - 1)
    [x_max, y_max] = np.int32(result_pts.max(axis=0).ravel() + 1)

    # ret = cv2.drawMatchesKnn(img1,kp1,img2,kp2,[goods],None)

    # 手动构造平移矩阵
    M = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    result = cv2.warpPerspective(img1, M.dot(H), (x_max - x_min, y_max - y_min) )  # 对img1进行平移和透视操作

    print("res shape:",result.shape)
    print("img1 shape:",img1.shape)
    print("img2 shape:",img2.shape)

    result[-y_min:-y_min + h2, -x_min:-x_min + w2] = img2  # 把img2放进来(因为img1变换后的矩阵也平移了，所以img2也要做对应的平移)
    cv2.imshow('result222222', result)
    top, bot, left, right = 100, 100, 0, 500
    rows, cols = img1.shape[:2]
    print("img1:",img1.shape)
    for col in range(cols):
        if img1[:, col].any() and result[:, col].any():
            left = col
            break
    for col in range(cols - 1, 0, -1):
        if img1[:, col].any() and result[:, col].any():
            right = col
            break

    print("left:",left)
    print("right:",right)
    res = np.zeros([rows, cols, 3], np.uint8)
    for row in range(rows):
        for col in range(cols):
            if not img1[row, col].any():
                res[row, col] = result[row, col]
            elif not result[row, col].any():
                res[row, col] = img1[row, col]
            else:
                img1Len = float(abs(col - left))
                testImgLen = float(abs(col - right))
                alpha = img1Len / (img1Len + testImgLen)
                res[row, col] = np.clip(img1[row, col] * (1 - alpha) + result[row, col] * alpha, 0, 255)

    # result[0:h2,0:w2] = img2
    # cv2.imshow('direct',direct)
    # cv2.imshow('result',result)
    # cv2.imshow('ret',ret)
    cv2.imshow('res', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def optimize_seam(self,img1,img2):
     pass





if __name__ == '__main__':
     # house_img_check()
     # book_trans_form()
     # image_search()
     # change_ad_img()
     # image_search()
     img_cpncat()


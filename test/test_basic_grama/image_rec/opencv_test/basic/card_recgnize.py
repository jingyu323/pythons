import cv2
import numpy as np
from imutils import contours

from opencv_test.basic.Stitcher import Stitcher
from opencv_test.basic.Stitcher2 import Stitcher2

"""
1.暴力匹配

2.bf.knnMatch()
 bf.knnMatch() , knn匹配过程中很可能发生错误的匹配，错误的匹配主要有两种：  匹配的特征点事错误的;图像上的特征点无法匹配。
 KNNMatch()：暴力法的基础上添加比率测试。
 
 
 3. FLANN匹配法
 快速最近邻搜索算法寻找的简称，它包含一组算法，这些算法针对大型数据集中的快速最近邻搜索和高维特征进行了优化。对于大型数据集，它的运行速度比BFMatcher快。
 
 
特征匹配：FLANN效果更好一些。
"""


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def show_img(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]  #用一个最小的矩形，把找到的形状包起来x,y,h,w
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    return cnts, boundingBoxes


def card_recgnize():
    tmplate = cv2.imread('../image/tmplate.jpg')
    FIRST_NUMBER = {
        "3": "American Express",
        "4": "Visa",
        "5": "MasterCard",
        "6": "Discover Card"
    }
    tm_gray = cv2.cvtColor(tmplate, cv2.COLOR_BGR2GRAY)
    # 二值化 灰度化之后
    tm_ref = cv2.threshold(tm_gray, 10, 255, cv2.THRESH_BINARY_INV)[1]

    counters, her = cv2.findContours(tm_ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for count in counters:
        cv2.drawContours(tmplate, [count], -1, (0, 0, 255), -1)

    show_img("tmplate", tmplate)

    refCnts = sort_contours(counters, method="left-to-right")[0]  # 排序，从左到右，从上到下
    digits = {}

    # 遍历每一个轮廓
    for (i, c) in enumerate(refCnts):
        # 计算外接矩形并且resize成合适大小
        (x, y, w, h) = cv2.boundingRect(c)
        roi = tm_ref[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))

        # 每一个数字对应每一个模板
        digits[i] = roi

    # 初始化卷积核
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    card = cv2.imread('../image/card3.png')
    card_gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)

    image = resize(card_gray, width=300)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 礼帽操作，突出更明亮的区域
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, rectKernel)
    show_img('tophat', tophat)
    #
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,  # ksize=-1相当于用3*3的
                      ksize=-1)

    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")

    print(np.array(gradX).shape)
    show_img('gradX', gradX)

    # 通过闭操作（先膨胀，再腐蚀）将数字连在一起
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    show_img('gradX', gradX)
    # THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
    thresh = cv2.threshold(gradX, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    show_img('thresh', thresh)

    # 再来一个闭操作
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)  # 再来一个闭操作
    show_img('thresh', thresh)

    # 计算轮廓

    threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
    cnts = threshCnts
    cur_img = image.copy()
    cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
    show_img('img', cur_img)
    locs = []

    # 遍历轮廓
    for (i, c) in enumerate(cnts):
        # 计算矩形
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
        if ar > 2.5 and ar < 4.0:

            if (w > 40 and w < 55) and (h > 10 and h < 20):
                # 符合的留下来
                locs.append((x, y, w, h))

    # 将符合的轮廓从左到右排序
    locs = sorted(locs, key=lambda x: x[0])
    output = []

    # 遍历每一个轮廓中的数字
    for (i, (gX, gY, gW, gH)) in enumerate(locs):
        # initialize the list of group digits
        groupOutput = []

        # 根据坐标提取每一个组
        group = image[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
        show_img('group', group)
        # 预处理
        group = cv2.threshold(group, 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        show_img('group', group)
        # 计算每一组的轮廓
        digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
        digitCnts = contours.sort_contours(digitCnts,
                                           method="left-to-right")[0]

        # 计算每一组中的每一个数值
        for c in digitCnts:
            # 找到当前数值的轮廓，resize成合适的的大小
            (x, y, w, h) = cv2.boundingRect(c)
            roi = group[y:y + h, x:x + w]
            roi = cv2.resize(roi, (57, 88))
            show_img('roi', roi)

            # 计算匹配得分
            scores = []

            # 在模板中计算每一个得分
            for (digit, digitROI) in digits.items():
                # 模板匹配
                result = cv2.matchTemplate(roi, digitROI,
                                           cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores.append(score)

            # 得到最合适的数字
            groupOutput.append(str(np.argmax(scores)))

        # 画出来
        cv2.rectangle(image, (gX - 5, gY - 5),
                      (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
        cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        # 得到结果
        output.extend(groupOutput)

    # 打印结果
    print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
    print("Credit Card #: {}".format("".join(output)))
    cv2.imshow("Image", image)
    cv2.waitKey(0)


def cv_show(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def tmp_match():
    template = cv2.imread('../image/bird.png')  # 读取灰度图
    cv_show('img', template)  # 展示图象
    img = cv2.imread('../image/the_bird.png')
    cv_show('img', img)

    # 获取小图像的高和宽
    h, w = template.shape[:2]

    # 不同的方法模板匹配的方式不同
    methodology = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF,
                   cv2.TM_SQDIFF_NORMED]

    # cv2.TM_CCOEFF：相关系数匹配。该方法计算输入图像和模板图像之间的相关系数，值越大表示匹配程度越好。
    # cv2.TM_CCOEFF_NORMED：标准归一化相关系数匹配。该方法计算输入图像和模板图像之间的标准归一化相关系数，也就是相关系数除以两个图像各自的标准差的乘积。值越大表示匹配程度越好。
    # cv2.TM_CCORR：相关性匹配。该方法计算输入图像和模板图像之间的相关性，值越大表示匹配程度越好。
    # cv2.TM_CCORR_NORMED：标准归一化相关性匹配。该方法计算输入图像和模板图像之间的标准归一化相关性，也就是相关性除以两个图像各自的标准差的乘积。值越大表示匹配程度越好。
    # cv2.TM_SQDIFF：平方差匹配。该方法计算输入图像和模板图像之间的平方差，值越小表示匹配程度越好。
    # cv2.TM_SQDIFF_NORMED：标准归一化平方差匹配。该方法计算输入图像和模板图像之间的标准归一化平方差，也就是平方差除以两个图像各自的标准差的乘积。值越小表示匹配程度越好。

    method = methodology[1]  # 选了一个cv2.TM_CCOEFF_NORMED方法进行图像匹配，匹配方式为比较模板和图像中各个区域的标准归一化相关系数
    res = cv2.matchTemplate(img, template, method)  # 比较完成的结果存储在res中，是一个ndarray类型的

    # 获取匹配结果中的最大值和最小值
    # 通过左上角点的位置，加上之前获取的小图像的宽和高h,w，就可以把图像在原图中的那个位置框出来了
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  # 最主要的是知道图像位置_loc

    # 不同的匹配算法，计算匹配到的位置的方式不同
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    # 通过左上角点的坐标和h，w计算右下角点的坐标
    bottom_right = (top_left[0] + w, top_left[1] + h)  # [0]为横坐标，[1]为纵坐标，横加宽纵加高就是右下角的点坐标

    # 绘制矩形
    resoult_img = cv2.rectangle(img.copy(), top_left, bottom_right, 255, 1)

    cv_show('img', resoult_img)


# 详细介绍
# https://blog.csdn.net/m0_50317149/article/details/130160067
def mul_tm_match():
    template = cv2.imread('../image/start.png')  # 读取灰度图目标
    img = cv2.imread('../image/stars.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 获取小图像的高和宽
    h, w = template.shape[:2]
    method = cv2.TM_CCOEFF_NORMED
    # 调用cv2.matchTemplate()函数进行图像匹配，比较完成的结果存储在res中，是一个ndarray类型的。具体来说就是原图中每一块区域所有像素点的比较结果，都存储在这个矩阵里。
    res = cv2.matchTemplate(img, template, method)
    # 与之前直接读取最大最小值不同，此次我们需要的是res中多个目标的结果，所以在此设置一个阈值
    threshold = 0.8
    loc = np.where(res >= threshold)  # 阈值为0.8，即取百分之80匹配的

    for loc in zip(*loc[::-1]):
        bottom_right = (loc[0] + w, loc[1] + h)
        cv2.rectangle(img, loc, bottom_right, (0, 0, 255), 2)

    cv_show('resoult', img)


def conor_dec():
    # 读取待检测的图像
    img = cv2.imread('../image/gezi.png')
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    # 调用函数 cornerHarris，检测角点，其中参数 2 表示 Sobel 算子的孔径大小，23 表示 Sobel 算子的孔径大小，0.04 表示 Harris 角点检测方程中的 k 值
    dst = cv2.cornerHarris(gray, 2, 23, 0.05)

    dst = cv2.dilate(dst, None)
    # 将检测到的角点标记出来
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv2.imshow('dst', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sift_dec():
    img1 = cv2.imread('../image/book.png', 0)
    img2 = cv2.imread('../image/books.png', 0)
    cv_show('img1', img1)
    cv_show('img2', img2)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # crossCheck表示两个特征点要互相匹，例如A中的第i个特征点与B中的第j个特征点最近的，并且B中的第j个特征点到A中的第i个特征点也是
    # NORM_L2: 归一化数组的(欧几里德距离)，如果其他特征计算方法需要考虑不同的匹配计算方式
    bf = cv2.BFMatcher(crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
    cv_show('img3', img3)


#  每进行特征点匹配 导致图像不能完全匹配
def img_concat():
    # 读取两张图片
    img1 = cv2.imread('../image/left.png')
    img2 = cv2.imread('../image/right.png')

    # 初始化ORB检测器
    orb = cv2.ORB_create()

    # 检测ORB特征点并计算描述符
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 创建暴力匹配器
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 进行匹配
    matches = bf.match(des1, des2)

    # 根据距离排序，距离小的是更好的匹配点
    matches = sorted(matches, key=lambda x: x.distance)

    # 绘制前N个匹配点
    N = 10  # 可以调整这个值来看到不同的拼接效果
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:N], None, flags=2)

    # 展示结果
    cv2.imshow('Matches', matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 我们的全景拼接算法将包含以下四个步骤
# 1：检测关键点（DoG，Harris等），并从两个输入图像中提取局部不变描述符（SIFT，SURF等）。
# 2：在两个图像之间匹配描述符。
# 3：使用RANSAC算法通过匹配的特征向量估计单应矩阵（或者叫变换矩阵）（homography matrix ）。
# 4：用 step #3 中的单应矩阵进行透视变
"""
  （1）特征点检测与图像匹配（stitching_match：Features Finding and Images Matching）
（2）计算图像间的变换矩阵（stitching_rotation：Rotation Estimation
（3）自动校准（stitching_autocalib Autocalibration）
（4）图像变形（stitching_warp Images Warping）
（5）计算接缝（stitching_seam：Seam Estimation）
（6）补偿曝光（stitching_exposure：Exposure Compensation）
（7）图像融合（stitching_blend：Image Blenders）
"""


def img_concat2():
    # 读取两张图片
    imageA = cv2.imread('../image/left.png')
    imageB = cv2.imread('../image/right.png')
    # 把图片拼接成全景图
    stitcher = Stitcher()
    print(imageA.shape, imageB.shape)

    (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)



    # 显示所有图片
    cv2.imshow("Image A", imageA)
    cv2.imshow("Image B", imageB)
    cv2.imshow("Keypoint Matches", vis)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def img_concat3():
    # 读取两张图片
    imageB = cv2.imread('../image/sitch/IMG_1786-2.jpg')
    imageA = cv2.imread('../image/sitch/IMG_1787-2.jpg')
    # 把图片拼接成全景图
    stitcher = Stitcher2()
    print(imageA.shape, imageB.shape)

    (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
    # 显示所有图片
    cv2.imshow("Image A", imageA)
    cv2.imshow("Image B", imageB)
    cv2.imshow("Keypoint Matches", vis)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def swicher_concat_img():
    img1 = cv2.imread('../image/left.png')
    img2 = cv2.imread('../image/right.png')

    print(img1.shape)
    print(img2.shape)

    # 创建 Stitcher 对象
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    # 调用 stitch 方法进行拼接
    result, stitched_image = stitcher.stitch([img1, img2])
    print(result)
    if result == cv2.Stitcher_OK:
        # 拼接成功，将结果保存到文件
        cv2.imshow('result.jpg', stitched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # 拼接失败
        print("拼接失败")


def knn_demo():
    img1 = cv2.imread('../image/bird.png')  # 读取灰度图
    img2 = cv2.imread('../image/the_bird.png')

    detector = cv2.ORB.create()  # Oriented FAST and Rotated BRIEF
    kp1 = detector.detect(img1, None)
    kp2 = detector.detect(img2, None)
    kp1, des1 = detector.compute(img1, kp1)  # keypoint 是关键点的列表,desc 检测到的特征的局部图的列表
    kp2, des2 = detector.compute(img2, kp2)

    # 创建BFMatcher对象
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(des1, des2, k=2)

    MIN_MATCH_COUNT = 4
    # 将距离从小到大排序
    # matches = sorted(matches, key=lambda x: x.distance)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2,[ good], None, flags=2)
    cv_show('img3', img3)
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        print(img1.shape)

        h,w= img1.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        cv_show('img2', img2)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # 画出绿色匹配线
                       singlePointColor=None,
                       matchesMask=matchesMask,  # 只画内点
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    cv_show('img33', img3)



#  flann 最终都是用knn 处理
def flnn_demo():
    img1 = cv2.imread('../image/bird.png')  # 读取灰度图
    img2 = cv2.imread('../image/the_bird.png')

    #  新版本直接使用 cv2.SIFT_create() 无需cv2.xfeatures2d.SIFT_create()
    sift = cv2.SIFT_create()
    # 查找监测点和匹配符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    """
    keypoint是检测到的特征点的列表
    descriptor是检测到特征的局部图像的列表
    """

    FLANN_INDEX_KDTREE = 0
    # 参数1：indexParams
    #    对于SIFT和SURF，可以传入参数index_params=dict(algorithm=FLANN_INDEX_KDTREE, trees=5)。
    #    对于ORB，可以传入参数index_params=dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12）。
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    # 参数2：searchParams 指定递归遍历的次数，值越高结果越准确，但是消耗的时间也越多。
    searchParams = dict(checks=50)

    # 使用FlannBasedMatcher 寻找最近邻近似匹配
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    # 使用knnMatch匹配处理，并返回匹配matches
    matches = flann.knnMatch(des1, des2, k=2)
    # 通过掩码方式计算有用的点
    matchesMask = [[0, 0] for i in range(len(matches))]

    # 通过描述符的距离进行选择需要的点
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:  # 通过0.7系数来决定匹配的有效关键点数量
            matchesMask[i] = [1, 0]

    drawPrams = dict(matchColor=(0, 255, 0),
                     singlePointColor=(255, 0, 0),
                     matchesMask=matchesMask,
                     flags=0)
    # 匹配结果图片
    img33 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **drawPrams)
    cv_show('img33', img33)

    MIN_MATCH_COUNT = 4
    # 将距离从小到大排序
    # matches = sorted(matches, key=lambda x: x.distance)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2,[ good], None, flags=2)
    cv_show('img3', img3)
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        #  计算单性适应性矩阵
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        print(img1.shape)

        h,w= img1.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        cv_show('img2', img2)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # 画出绿色匹配线
                       singlePointColor=None,
                       matchesMask=matchesMask,  # 只画内点
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    cv_show('img33', img3)


if __name__ == '__main__':
    # flnn_demo()
    # img_concat()
    # img_concat()
    # img_concat2()
    img_concat3()
    # swicher_concat_img()

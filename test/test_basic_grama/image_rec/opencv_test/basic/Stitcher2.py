import cv2
import numpy as np


class Stitcher2:
    def stitch(self, images, ratio=0.75, reprojThresh=5.0,showMatches=False):
        (imageA, imageB) = images
        # 检测A、B图片的SIFT关键特征点，并计算特征描述子
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)
        print("A shape:",imageA.shape)
        print("imageB shape:",imageB.shape)

        # 匹配两张图片的所有特征点，返回匹配结果
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # 如果返回结果为空，没有匹配成功的特征点，退出算法
        if M is None:
            return None

        # 否则，提取匹配结果
        # H是3x3视角变换矩阵
        (matches, H, status) = M

        # 获取原始图像的高宽
        h1, w1 = imageA.shape[:2]
        h2, w2 = imageB.shape[:2]
        # 获取两幅图的边界坐标
        img1_pts = np.float32([[0, 0], [0, h1 - 1], [w1 - 1, h1 - 1], [w1 - 1, 0]]).reshape(-1, 1, 2)
        img2_pts = np.float32([[0, 0], [0, h2 - 1], [w2 - 1, h2 - 1], [w2 - 1, 0]]).reshape(-1, 1, 2)

        # 获取img1的边界坐标变换之后的坐标
        img1_transform = cv2.perspectiveTransform(img1_pts, H)
        print('img1_pts',img1_pts)
        print('img2_pts',img2_pts)
        print('img1_transform',img1_transform)
        # 把img2和转换后的边界坐标连接起来
        result_pts = np.concatenate((img2_pts, img1_transform), axis=0)
        print("result_pts",result_pts)
        # result_pts.min(axis=0)  #对行聚合，返回每一列的最小值
        print(np.int32(result_pts.min(axis=0).ravel() - 1))
        print( np.int32(result_pts.max(axis=0).ravel() + 1))
        [x_min, y_min] = np.int32(result_pts.min(axis=0).ravel() - 1)
        [x_max, y_max] = np.int32(result_pts.max(axis=0).ravel() + 1)

        # ret = cv2.drawMatchesKnn(img1,kp1,img2,kp2,[goods],None)

        # 手动构造平移矩阵
        M = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

        print("x_max - x_min :",x_max - x_min)
        print("x_max   :",x_max  )
        print("y_max - y_min :",y_max - y_min)
        print("y_max   :",y_max )

        print("M.dot(H)",M.dot(H))
        result = cv2.warpPerspective(imageA, M.dot(H), (x_max,y_max  ))  # 对img1进行平移和透视操作
        self.cv_show('imageAAAA999', result)
        print("H:", H)

        # 将图片A进行视角变换，result是变换后图片 ，因为未添加平移变换 导致有被截图的情况
        # result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        # self.cv_show('resul555555', result)


        res = self.optimize_seam(  imageB, result,-y_min,-x_min)
        self.cv_show('res4444444444444444', res)

        # imageB = cv2.resize(imageB, (imageB.shape[1], result.shape[0]))
        print("resul555555 shape:",result.shape)


        # # 将图片B传入result图片最左端

        result[-y_min:-y_min + h2, -x_min:-x_min + w2] = imageB
        # result[0-y_min: h2-y_min, w1:w1 + w2 ] = imageB
        # result[0:imageA.shape[0], 0+imageA.shape[1]:imageB.shape[1]+imageA.shape[1]] = imageA
        self.cv_show('result2332', result)

        # 检测是否需要显示图片匹配
        if showMatches:
            # 生成匹配图片
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            # 返回结果
            return (result, vis,res)

        # 返回匹配结果
        return result

    def cv_show(self, name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#
# detectAndDescribe 方法分别检测两张图片的 SIFT 特征点和计算描述子
# 使用 matchKeypoints 方法对特征点进行匹配。
# 判断匹配是否成功；若成功，使用单应性矩阵 H 通过透视变换对其中一张图片进行变换，然后与另一张图片合并
#
    def detectAndDescribe(self, image):
        # 将彩色图片转换成灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 建立SIFT生成器
        # descriptor = cv2.SIFT_create()
        descriptor = cv2.SIFT_create()
        # 检测SIFT特征点，并计算描述子
        (kps, features) = descriptor.detectAndCompute(gray, None)

        # 将结果转换成NumPy数组
        kps = np.float32([kp.pt for kp in kps])
        # 返回特征点集，及对应的描述特征
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # 建立暴力匹配器
        # matcher = cv2.BFMatcher()
        #
        # # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2
        # rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # 对描述子进行特征匹配
        rawMatches = flann.knnMatch(featuresA, featuresB, k=2)  # 用的knnmatch匹配


        matches = []
        for m in rawMatches:
            # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
            if  m[0].distance < m[1].distance * ratio:

                # 存储两个点在featuresA, featuresB中的索引值
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # 当筛选后的匹配对大于4时，计算视角变换矩阵
        if len(matches) > 4:
            # 获取匹配对的点坐标
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # 计算视角变换矩阵
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 5)

            # 返回结果
            return (matches, H, status)

        # 如果匹配对小于4时，返回None
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # 初始化可视化图片，将A、B图左右连接到一起
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # 联合遍历，画出匹配对
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # 当点对匹配成功时，画到可视化图上
            if s == 1:
                # 画出匹配对
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # 返回可视化结果
        return vis


    def optimize_seam(self,srcImg,warpImg,y,x):
        print("srcImg.shape:",srcImg.shape)
        rows, cols = srcImg.shape[:2]
        left=0
        right=0
        #  求出重叠区域左右值
        for col in range(cols):
            if srcImg[:, col].any() and warpImg[:, col].any():
                print("srcImg[:, col] :", srcImg[:, col],col,len(srcImg[:, col]))
                print("warpImg[:, col]:", warpImg[:, col],col,len(warpImg[:, col]))
                left = col
                break
        for col in range(cols - 1, 0, -1):
            if srcImg[:, col].any() and warpImg[:, col].any():
                right = col
                break
        print("left:",left)
        print("right:",right)
        res = warpImg.copy()
        for row in range(rows):
            for col in range(cols):
                if not srcImg[row, col].any():
                    res[row+y, col+x] = warpImg[row+y, col+x]
                elif not warpImg[row+y, col+x].any():
                    res[row+y, col+x] = srcImg[row, col]
                else:
                    srcImgLen = float(abs(col - left))
                    testImgLen = float(abs(col - right))
                    alpha = srcImgLen / (srcImgLen + testImgLen)
                    res[row+y, col+x] = np.clip(srcImg[row, col] * (1 - alpha) + warpImg[row+y, col+x] * alpha, 0, 255)

        return  res






import cv2
import numpy as np


class Stitcher:
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

        self.cv_show('imageAAAAAAAAAAAAAAA', imageA)
        print("H:", H)

        # 将图片A进行视角变换，result是变换后图片
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        self.cv_show('resul555555', result)
        M = np.float32([[1, 0, 30], [0, 1, 10]])
        result = cv2.warpAffine(result, M, dsize=(result.shape[1], result.shape[0]))
        self.cv_show('resul6666666666', result)


        imageB = cv2.resize(imageB, (imageB.shape[1], imageA.shape[0]))
        print("resul555555 shape:",result.shape)


        # # 将图片B传入result图片最左端
        result[0:imageA.shape[0], 0+imageA.shape[1]:imageB.shape[1]+imageA.shape[1]] = imageB
        # result[0:imageA.shape[0], 0+imageA.shape[1]:imageB.shape[1]+imageA.shape[1]] = imageA
        self.cv_show('result2332', result)
        # 检测是否需要显示图片匹配
        if showMatches:
            # 生成匹配图片
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            # 返回结果
            return (result, vis)

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

        # 计算特征点和描述子

        # 建立SIFT生成器
        # descriptor = cv2.SIFT_create()
        descriptor = cv2.SIFT_create()
        # 检测SIFT特征点，并计算描述子
        (kps, features) = descriptor.detectAndCompute(gray, None)


        # 将结果转换成NumPy数组
        kps = np.float32([kp.pt for kp in kps])
        print("kps:", kps)

        # 返回特征点集，及对应的描述特征
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # 建立暴力匹配器
        # matcher = cv2.BFMatcher()
        #
        # # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2
        # rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

        # 创建特征匹配器
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # 对描述子进行特征匹配
        rawMatches = flann.knnMatch(featuresA, featuresB, k=2)  # 用的knnmatch匹配

        print("rawMatches:", rawMatches)

        matches = []
        for m,n in rawMatches:
            # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
            if  m.distance < n.distance * 0.75:
                print(m.distance, n.distance)

                # 存储两个点在featuresA, featuresB中的索引值
                # matches.append((m[0].trainIdx, m[0].queryIdx))
                matches.append(m)

        # 当筛选后的匹配对大于4时，计算视角变换矩阵
        if len(matches) >= 4:
            # 获取匹配对的点坐标
            # ptsA = np.float32([kpsA[i] for (_, i) in matches])
            # ptsB = np.float32([kpsB[i] for (i, _) in matches])
            src_points = np.float32([kpsA[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
            des_points = np.float32([kpsB[m.trainIdx] for m in matches]).reshape(-1, 1, 2)


            # 计算视角变换矩阵
            # (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            # src_points = np.float32([kpsA[i] for (_, i) in matches])
            # des_points = np.float32([kpsB[i] for  (i, _) in matches])

            H, status = cv2.findHomography(src_points, des_points, cv2.RANSAC, 5)  # 参数5表示：允许有5个关键点的误差

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
        # for ((trainIdx, queryIdx), s) in zip(matches, status):
        #     # 当点对匹配成功时，画到可视化图上
        #     if s == 1:
        #         # 画出匹配对
        #         ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
        #         ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
        #         cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # 返回可视化结果
        return vis
import cv2
import numpy as np


def four_point_transfrom(img, pts):
    newPoints = []
    sumPoints = []
    subPoints = []
    for x, y in pts:
        sumPoints.append(x + y)
        subPoints.append(x - y)

    # get index
    newPoints.append(pts[(np.argmin(sumPoints))])  # 0-A

    newPoints.append(pts[(np.argmax(subPoints))])  # 1-B
    newPoints.append(pts[(np.argmin(subPoints))])  # 2-C

    newPoints.append(pts[(np.argmax(sumPoints))])  # 3-D

    return newPoints
def preprocess(img):
    # 灰度转换
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #高斯滤波
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)

    cv2.imshow('imgBlur', imgBlur)
    cv2.waitKey()
    #边缘检测
    imgEdge = cv2.Canny(imgBlur, 75, 200)
    cv2.imshow('Canny', imgEdge)
    cv2.waitKey()
	#返回预处理的结果
    return imgEdge

def getCnts(img):
    cnts, hier = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, cnts, -1, (0, 0, 255), 3)




    print(len(cnts))
    docCnts = None
    # 确保有轮廓
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            print(c)
            # 轮廓近似
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                docCnts = approx
                break

        return docCnts
def getWrap(img, wraped, w, h):
    src = np.array(wraped, dtype="float32")
    dst = np.array([[0, 0],
                    [w - 1, 0],
                    [0, h - 1],
                    [w - 1, h - 1]], dtype="float32")

    # src and dst must be type of float32
    M = cv2.getPerspectiveTransform(src, dst)

    imgRes = cv2.warpPerspective(img, M, (w, h))

    return imgRes
def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes


def main():
    ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}
    # read img
    imgOriginal = cv2.imread('../image/exam.jpg')
    imgOriginal = cv2.resize(imgOriginal, (500, 600))
    w = imgOriginal.shape[0]
    h = imgOriginal.shape[1]
    cv2.imshow('imgOriginal', imgOriginal)
    cv2.waitKey()
    drawImg = imgOriginal.copy()

    # preprocess
    imgPre = preprocess(drawImg)

    # contours
    cur_cnts = getCnts(imgPre)

    # imgPerspective
    wraped = four_point_transfrom(imgPre, cur_cnts.reshape(4, 2))
    imgWrap = getWrap(imgOriginal, wraped, w, h)
    cv2.imshow('imgWrap', imgWrap)
    cv2.waitKey()

    # threshold
    imgWrapGray = cv2.cvtColor(imgWrap, cv2.COLOR_BGR2GRAY)
    imgThreshold = cv2.threshold(imgWrapGray, 20, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cv2.imshow("Wraped and threshold", imgThreshold)
    cv2.waitKey()

    # find Contours2

    draw_cnts = imgWrap.copy()
    thresh_cnts,hers = cv2.findContours(imgThreshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(draw_cnts, thresh_cnts, -1, (0, 0, 255), 3)

    cv2.imshow("thresh_cnts", draw_cnts)
    cv2.waitKey()

    questionCnts = []

    # 过滤出圆形轮廓
    # 遍历
    for c in thresh_cnts:
        # 计算比例和大小
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)


        # 根据实际情况指定标准
        if w >= 50 and h >= 40 and ar >= 0.9 and ar <= 1.6:
            print('w=', w)# 1.(24,15)
            print('h=', h)
            print('---------------')
            questionCnts.append(c)

    # 对于同一个题，x相同，y不同
    # 对于不同题，x不同，y相同

    questionCnts = sort_contours(questionCnts, method="top-to-bottom")[0]
    correct = 0
    count = 0

    # 每排有5个选项
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
        # 排序-按y坐标，每次排一组，一组为5个
        cnts = sort_contours(questionCnts[i:i + 5])[0]
        bubbled = None
        # 遍历每一组结果
        for (j, c) in enumerate(cnts):
            # 使用mask来判断结果
            mask = np.zeros(imgThreshold.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)  # -1表示填充，[c]表示当前选项的位置

            # 通过计算非零点数量来算是否选择这个答案
            mask = cv2.bitwise_and(imgThreshold, imgThreshold, mask=mask)#相当于只保留了白色的部分
            total = cv2.countNonZero(mask)
            print('total=',total)
            #判断一下，如果他涂了两个选项，应该另外处理
            if total>1000:
                count += 1
            if count< 2:
                # 通过阈值判断，如果是第一次/当前值比上一次的大，则bubbled记录下来，直到最后bubbled就是一组中值最大的那个选项
                if bubbled is None or total > bubbled[0]:
                    bubbled = (total, j)
            else:
                print('题目[%d]选项大于1'.format(j))
        print('-------------------')
        count=0


        # 对比正确答案
        color = (0, 0, 255)
        k = ANSWER_KEY[q]

        # 判断正确,bubbled=(656,1),其中第二个依次表示A,B,C,D,E，
        if k == bubbled[1]:
            color = (0, 255, 0)
            correct += 1

        # 绘图
        cv2.drawContours(imgWrap, [cnts[k]], -1, color, 3)

    score = (correct / 5.0) * 100
    print("[INFO] score: {:.2f}%".format(score))
    cv2.putText(imgWrap, "{:.2f}%".format(score), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imshow("Original", imgOriginal)
    cv2.imshow("Exam", imgWrap)
    cv2.waitKey()
if __name__ == '__main__':
    main()
import cv2
import numpy as np


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

def  show_img(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts] #用一个最小的矩形，把找到的形状包起来x,y,h,w
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    return cnts, boundingBoxes
def card_recgnize():
    tmplate = cv2.imread('../image/tmplate.jpg')

    tm_gray = cv2.cvtColor(tmplate, cv2.COLOR_BGR2GRAY)
    # 二值化 灰度化之后
    tm_ref = cv2.threshold(tm_gray, 10, 255, cv2.THRESH_BINARY_INV)[1]

    counters,her =cv2.findContours(tm_ref,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for count in counters:
        cv2.drawContours(tmplate,[count],-1,(0,0,255),-1)

    show_img("tmplate", tmplate)

    refCnts =  sort_contours(counters, method="left-to-right")[0]  # 排序，从左到右，从上到下
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

    card = cv2.imread('../image/card.png')
    card_gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)

    image =  resize(card_gray, width=300)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 礼帽操作，突出更明亮的区域
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    show_img('tophat', tophat)
    #
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,  # ksize=-1相当于用3*3的
                      ksize=-1)

    gradX = np.absolute(gradX)

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
        group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
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


    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



if __name__ == '__main__':
    card_recgnize()

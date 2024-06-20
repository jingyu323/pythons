import cv2  # 答题卡识别
import numpy as np
import pandas as pd

from opencv_test.utils.cv2_related import cv_show

def judgeX(x,mode):
    if mode=="point":
        if x<600:
            return int(x/100)+1
        elif x>600 and x<1250:
            return int((x-650)/100)+6
        elif x>1250 and x<1900:
            return int((x-1250)/100)+11
        elif x>1900:
            return int((x-1900)/100)+16
    elif mode=="ID":
        return int((x-110)/260)+1
    elif mode=="subject":
        if x<1500:
            return False


def judgeY(x,mode):
    if mode=="point":
        if x<600:
            return int(x/100)+1
        elif x>600 and x<1250:
            return int((x-650)/100)+6
        elif x>1250 and x<1900:
            return int((x-1250)/100)+11
        elif x>1900:
            return int((x-1900)/100)+16
    elif mode=="ID":
        return int((x-110)/260)+1
    elif mode=="subject":
        if x<1500:
            return False

def imgBrightness(img1, c, b):
    rows, cols = img1.shape
    blank = np.zeros([rows, cols], img1.dtype)
    rst = cv2.addWeighted(img1, c, blank, 1 - c, b)
    return rst
def judge(x,y,mode):
    if judgeY(y,mode)!=False and judgeX(x,mode)!=False:
        if mode=="point":
           return (int(y/560)*20+judgeX(x,mode),judgeY(y,mode))
        elif mode=="ID":
           return (judgeX(x,mode),judgeY(y,mode))
        elif mode=="subject":
           return judgeY(y,mode)
    else:
      return 0
def judge_point(answers,mode):
    IDAnswer=[]
    for answer in answers:
        if(judge(answer[0],answer[1],mode)!=0):
          IDAnswer.append(judge(answer[0],answer[1],mode))
        else:
          continue
    IDAnswer.sort()
    return IDAnswer
def judge_ID(IDs,mode):
    student_ID=[]
    for ID in IDs:
        if(judge(ID[0],ID[1],mode)!=False):
          student_ID.append(judge(ID[0],ID[1],mode))
        else:
          continue
    student_ID.sort()
    return student_ID
def judge_Subject(subject,mode):
    return judge(subject[0][0],subject[0][1],mode)


def order_points(pts):
    # 初始化坐标点
    rect = np.zeros((4, 2), dtype="float32")

    # 获取左上角和右下角坐标点
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 分别计算左上角和右下角的离散差值
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    # 获取坐标点，并将它们分离开来
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算新图片的宽度值，选取水平差值的最大值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # 计算新图片的高度值，选取垂直差值的最大值
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 构建新图片的4个坐标点
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 获取仿射变换矩阵并应用它
    M = cv2.getPerspectiveTransform(rect, dst)
    # 进行仿射变换
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # 返回变换后的结果
    return warped
def red_img():
    img = cv2.imread('../image/datika.png')

    # img = cv2.resize(img,(int (img.shape[1]*0.25), int(img.shape[0]*0.25)))
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯滤波
    # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # 增强亮度
    # blurred = imgBrightness(gray, 1.5, 3)

    # 自适应二值化
    # blurred = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 2)

    ret, blurred = cv2.threshold(gray, 10, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

    '''
    adaptiveThreshold函数：第一个参数src指原图像，原图像应该是灰度图。
        第二个参数x指当像素值高于（有时是小于）阈值时应该被赋予的新的像素值
        第三个参数adaptive_method 指： CV_ADAPTIVE_THRESH_MEAN_C 或 CV_ADAPTIVE_THRESH_GAUSSIAN_C
        第四个参数threshold_type  指取阈值类型：必须是下者之一  
                            • CV_THRESH_BINARY,
                            • CV_THRESH_BINARY_INV
        第五个参数 block_size 指用来计算阈值的象素邻域大小: 3, 5, 7, ...
        第六个参数param1    指与方法有关的参数。对方法CV_ADAPTIVE_THRESH_MEAN_C 和 CV_ADAPTIVE_THRESH_GAUSSIAN_C， 它是一个从均值或加权均值提取的常数, 尽管它可以是负数。
    '''
    # blurred = cv2.copyMakeBorder(blurred, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    edged = cv2.Canny(blurred, 10, 255)
    cnts, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    docCnt = []
    count = 0
    # 确保至少有一个轮廓被找到
    if len(cnts) > 0:
        # 将轮廓按照大小排序
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # 对排序后的轮廓进行循环处理
    for c in cnts:
        # 获取近似的轮廓
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.001 * peri, True)
        # 如果近似轮廓有四个顶点，那么就认为找到了答题卡
        if len(approx) == 4:
            docCnt.append(approx)
            # image_contours = cv2.drawContours(img, [approx], contourIdx=-1, color=(0, 255, 0), thickness=3)  # 绘制轮廓
            # cv_show('image_contours', image_contours)
            print(approx)
            count += 1
            # if count == 3:
            #     # break
    copy = img.copy()
    cv2.drawContours(copy, docCnt, -1, (0, 255, 0), 2)
    cv2.imshow('copy', copy)

    select_are=[(8,238
                 ),(510,238),(8,686),(510,686)]
    id_are=[(224,20
                 ),(421,20),(224,210),(421,210)]

    subject_are=[(447,20
                 ),(510,20),(447,210),(510,210)]

    #四点变换，划出选择题区域
    paper = four_point_transform(img,np.array(select_are).reshape(4,2))
    warped = four_point_transform(gray,np.array(select_are).reshape(4,2))
    #四点变换，划出准考证区域
    ID_Area = four_point_transform(img,np.array(id_are).reshape(4,2))
    ID_Area_warped = four_point_transform(gray,np.array(id_are).reshape(4,2))
    #四点变换，划出科目区域
    Subject_Area = four_point_transform(img,np.array(subject_are).reshape(4,2))
    Subject_Area_warped = four_point_transform(gray,np.array(subject_are).reshape(4,2))

    cv_show('paper', paper)
    cv_show('ID_Area', ID_Area)
    cv_show('Subject_Area', Subject_Area)
    cv_show('warped', warped)
    cv_show('ID_Area_warped', ID_Area_warped)
    cv_show('Subject_Area_warped', Subject_Area_warped)

    '''
        处理选择题区域统计答题结果
    '''
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # thresh = cv2.resize(thresh, (2400, 2800), cv2.INTER_LANCZOS4)
    # paper = cv2.resize(paper, (2400, 2800), cv2.INTER_LANCZOS4)
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(paper, cnts, -1, (0, 255, 0), 1)
    # cv_show('paper', paper)
    questionCnts = []
    answers = []
    # 对每一个轮廓进行循环处理
    for c in cnts:
        # 计算轮廓的边界框，然后利用边界框数据计算宽高比
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # 判断轮廓是否是答题框
        if w >= 10 and h >= 5 and ar >= 1 and ar <= 1.8:
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            questionCnts.append(c)
            answers.append((cX, cY))
            cv2.circle(paper, (cX, cY), 7, (255, 255, 255), -1)

    print("ID_Answer:", answers)
    ID_Answer = judge_point(answers, mode="point")

    # cv2.drawContours(paper, questionCnts, -1, (255, 0, 0), 3)

    '''
        读取结果
        '''
    df = pd.read_excel("answer.xlsx")
    index_list = df[["题号"]].values.tolist()

    true_answer_list = df[["答案"]].values.tolist()

    index = []
    true_answer = []
    score = 0
    # 去括号
    for i in range(len(index_list)):
        index.append(index_list[i][0])
    for i in range(len(true_answer_list)):
        true_answer.append(true_answer_list[i][0])

    answer_index = []
    answer_option = []

    print("ID_Answer:",ID_Answer)
    for answer in ID_Answer:
        answer_index.append(answer[0])
        answer_option.append(answer[1])
    for i in range(len(index)):
        if answer_option[i] == true_answer[i]:
            score += 1
        if i + 1 == len(answer_option):
            break







def hellotransform():
    image = cv2.imread('../image/hello01.png')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, template_img = cv2.threshold(gray, 10, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    cnts,her=cv2.findContours(template_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    screenCnt=[]
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            print(screenCnt)
            image_contours = cv2.drawContours(image, [screenCnt], contourIdx=-1, color=(0, 255, 0), thickness=3)  # 绘制轮廓
            cv_show('image_contours', image_contours)

    print(type(screenCnt))
    print(screenCnt.reshape(4,2))

    # 获取原始的坐标点 ， 使用reshape 进行行列变换
    pts = np.array( screenCnt.reshape(4,2), dtype="float32")

    print(pts)

    # 对原始图片进行变换
    warped = four_point_transform(image, pts)

    # 结果显示
    cv2.imshow("Original", image)
    cv2.imshow("Warped", warped)
    cv2.waitKey(0)



if __name__ == '__main__':
    red_img()
import sys

import cv2
import numpy as np
from imutils.video import FPS


def mul_tracker_test():
    # 初始化视频捕捉
    # video = cv2.VideoCapture(0)

    video = cv2.VideoCapture("../video/gaosu.mp4")
    multi_tracker = cv2.TrackerMIL()
    object_ids = []  # 假设这是从其他方式获取的对象ID列表

    ret, frame = video.read()
    if not ret:
        print("无法读取视频文件")
        exit()
    bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)

    # multi_tracker.init(frame, bbox)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        object_ids.append(bbox)
        print(bbox)
        # 检查是否有需要追踪的对象
        if len(object_ids) > 0:
            # 更新所有追踪器的位置
            success, boxes = multi_tracker.update(frame)
            print(boxes)

            # 画出追踪器的位置
            for i, newbox in enumerate(boxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

        # 在此处理帧中的其他对象或操作
        # ...

        # 显示结果
        cv2.imshow("Tracking", frame)

        # 按 'q' 退出循环
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #       break

    # 释放捕捉器资源
    video.release()
    cv2.destroyAllWindows()

def createTypeTracker(trackerType):
    trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT','VIT']
    if trackerType == trackerTypes[0]:
        tracker = cv2.legacy.TrackerBoosting().create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL().create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF().create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.legacy.TrackerTLD().create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.legacy.TrackerMedianFlow().create()
    elif trackerType == trackerTypes[5]:  # 暂时存在问题
        tracker = cv2.TrackerGOTURN().create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.legacy.TrackerMOSSE().create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT().create()
    elif trackerType == trackerTypes[8]:
        tracker = cv2.TrackerVit().create()
    else:
        tracker = None

    return tracker
def tracker_test():
    # 初始化视频捕捉
    # video = cv2.VideoCapture(0)

    video = cv2.VideoCapture("../video/gaosu.mp4")
    # 创建一个跟踪器，algorithm: KCF、CSRT、DaSiamRPN、GOTURM、MIL
    tracker_type = 'MIL'
    tracker = createTypeTracker(tracker_type)
    # 如果视频没有打开，退出。
    if not video.isOpened():
        "Could not open video"
        sys.exit()

    # 读第一帧。
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # 定义一个初始边界框
    bbox = (287, 23, 86, 320)
    # 取消注释下面的行以选择一个不同的边界框
    bbox = cv2.selectROI(frame, False)
    # 用第一帧和包围框初始化跟踪器
    ok = tracker.init(frame, bbox)
    while True:
        # 读取一个新的帧
        ok, frame = video.read()
        if not ok:
            break
        # 启动计时器
        timer = cv2.getTickCount()
        # 更新跟踪器
        ok, bbox = tracker.update(frame)
        # 计算帧率(FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # 绘制包围框
        if ok:
            # 跟踪成功
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # 跟踪失败
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
                        2)

        # 在帧上显示跟踪器类型名字
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
        # 在帧上显示帧率FPS
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
        # 显示结果
        cv2.imshow("Tracking", frame)

        # 按ESC键退出
        k = cv2.waitKey(1) & 0xff
        if k == 27: break
    cv2.destroyAllWindows()

def multi_tracker():
    # 初始化视频源
    video = cv2.VideoCapture("../video/gaosu.mp4")
    # 初始化目标的边框
    trackers=[]

    # 重复运行
    while True:
        ret, frame = video.read()
        if not ret:
            break

        timer = cv2.getTickCount()
        if len(trackers) >0:
            for tracker in trackers:
                success, bbox = tracker.update(frame)

                if success:
                    # 跟踪成功
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                cv2.putText(frame, tracker.Params.__name__+ " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50),
                            2);
                cv2.imshow('MultiTracker', frame)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50),
                    2);
        cv2.imshow('MultiTracker', frame)
        k = cv2.waitKey(100) & 0xFF
        if k == ord('s'):
            initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
            tracker = cv2.TrackerCSRT().create()
            tracker.init(frame, initBB)
            trackers.append(tracker)
        elif k == 27:
            break
    video.release()
    cv2.destroyAllWindows()


#  同时选中多个目标   没弄成功
def multi_tracker2():
    # 初始化视频源
    video = cv2.VideoCapture("../video/gaosu.mp4")
    # 初始化目标的边框
    trackers=[]

    # 重复运行
    while True:
        ret, frame = video.read()
        if not ret:
            break

        timer = cv2.getTickCount()
        if len(trackers) >0:
            for tracker in trackers:
                success, bbox = tracker.update(frame)

                if success:
                    # 跟踪成功
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                cv2.putText(frame, tracker.Params.__name__+ " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50),
                            2);
                cv2.imshow('MultiTracker', frame)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50),
                    2);
        cv2.imshow('MultiTracker', frame)
        k = cv2.waitKey(100) & 0xFF
        if k == ord('s'):
            ROIs = cv2.selectROIs("Frame", frame, fromCenter=False, showCrosshair=True)

            for ROI in ROIs:
                x, y, w, h = ROI
                roi = frame[int(y):int(y + h), int(x):int(x + w)]

                tracker = cv2.TrackerCSRT().create()
                # tracker.init(frame, roi)
                trackers.append(tracker)

    video.release()
    cv2.destroyAllWindows()


#    https://www.cnblogs.com/libai123456/p/17630025.html
# 这种算法的缺陷主要有两个：一是跟踪的目标被遮挡后，跟踪会丢失；二是如果不是第一帧出现的人物，不会被标记跟踪
def  tracker_demo1():
    # 读取视频
    cap = cv2.VideoCapture('vtest.avi')

    # 读取第一帧图片，提取其特征点向量
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # 角点（特征点）检测 （shi-Tomasi角点检测）
    old_pts = cv2.goodFeaturesToTrack(old_gray, maxCorners=100, qualityLevel=0.3, minDistance=10)
    # print(old_pts)

    # 创建一个mask
    mask = np.zeros_like(old_frame)
    # 随机颜色
    color = np.random.randint(0, 255, size=(100, 3))
    print(color[1].tolist())

    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 光流估计
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, old_pts, None, winSize=(15, 15), maxLevel=4)
        #     print(len(next_pts))

        # 哪些特征点找到了，哪些特征点没有找到
        good_new = next_pts[status == 1]
        good_old = old_pts[status == 1]
        print(good_new)

        # 绘制特征点的轨迹
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            x1, y1 = new
            x0, y0 = old
            mask = cv2.line(mask, (x1, y1), (x0, y0), color[i].tolist(), 2)
            frame = cv2.circle(frame, (x1, y1), 5, color[i].tolist(), -1)

        img = cv2.add(mask, frame)  # 把轨迹和当前帧图片融合

        cv2.imshow('mask', mask)
        cv2.imshow('video', img)
        key = cv2.waitKey(100)
        if key == ord('q'):
            break

        # 更新
        old_gray = gray.copy()
        old_pts = good_new.reshape(-1, 1, 2)  # 要把 good_new 的维度变回old_pts一样

    cap.release()
    cv2.destroyAllWindows()



def  tracker_demo2():
    # 读取视频
    cap = cv2.VideoCapture('cars.mp4')

    # 读取第一帧图片，提取其特征点向量
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # 创建sift对象  （SIFT关键点检测）
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=100)  # 只需要100个关键点，不指定nfeatures的话关键点太多
    # 进行检测
    kpoint = sift.detect(old_gray)
    old_pts = cv2.KeyPoint_convert(kpoint)  # 把关键点转换为坐标
    old_pts = old_pts.reshape(-1, 1, 2)  # 维度变换

    # 创建一个mask
    mask = np.zeros_like(old_frame)
    # 随机颜色
    color = np.random.randint(0, 255, size=(100, 3))
    print(color[1].tolist())

    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 光流估计
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, old_pts, None, winSize=(15, 15), maxLevel=4)
        #     print(len(next_pts))

        # 哪些特征点找到了，哪些特征点没有找到
        good_new = next_pts[status == 1]
        good_old = old_pts[status == 1]
        #     print(good_new)

        # 绘制特征点的轨迹
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            x1, y1 = new
            x0, y0 = old
            mask = cv2.line(mask, (x1, y1), (x0, y0), color[i].tolist(), 2)
            frame = cv2.circle(frame, (x1, y1), 5, color[i].tolist(), -1)

        img = cv2.add(mask, frame)

        cv2.imshow('mask', mask)
        cv2.imshow('video', img)
        key = cv2.waitKey(100)
        if key == ord('q'):
            break

        # 更新
        old_gray = gray.copy()
        old_pts = good_new.reshape(-1, 1, 2)  # 要把 good_new 的维度变回old_pts一样

    cap.release()
    cv2.destroyAllWindows()
def  tracker_demo3():
    # 定义opencv中的七种目标追踪算法
    OPENCV_OBJECT_TRACKERS = {
        'boosting': cv2.TrackerBoosting_create,
        'csrt': cv2.TrackerCSRT_create,
        'kcf': cv2.TrackerKCF_create,
        'mil': cv2.TrackerMIL_create,
        'tld': cv2.TrackerTLD_create,
        'medianflow': cv2.TrackerMedianFlow_create,
        'mosse': cv2.TrackerMedianFlow_create
    }

    trackers = cv2.MultiTracker_create()  # 创建MultiTracker对象

    cap = cv2.VideoCapture('D:/videos/los_angeles.mp4')

    while True:
        ret, frame = cap.read()

        if frame is None:
            break
        # 绘制追踪到的矩形区域（要在imshow之前）
        success, boxes = trackers.update(frame)

        for box in boxes:  # 显示追踪框
            (x, y, w, h) = [int(v) for v in box]  # box是浮点型，画图需要整型
            cv2.rectangle(frame, (x, y), (x + w, y + h), [0, 0, 255], 2)

        cv2.imshow('viedo', frame)

        key = cv2.waitKey(100)

        if key == ord('s'):  # 按s选择需要追踪的目标
            roi = cv2.selectROI('viedo', frame, showCrosshair=False)
            # 创建一个实际的目标追踪器
            tracker = OPENCV_OBJECT_TRACKERS['tld']()
            trackers.add(tracker, frame, roi)

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    multi_tracker2()

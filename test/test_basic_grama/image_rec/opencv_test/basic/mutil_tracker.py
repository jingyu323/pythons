import sys

import cv2


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
    trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
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
    else:
        tracker = None

    return tracker
def tracker_test():
    # 初始化视频捕捉
    # video = cv2.VideoCapture(0)

    video = cv2.VideoCapture("../video/gaosu.mp4")
    # 创建一个跟踪器，algorithm: KCF、CSRT、DaSiamRPN、GOTURM、MIL
    tracker_type = 'CSRT'
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


if __name__ == '__main__':
    tracker_test()

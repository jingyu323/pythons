import cv2  #  需要安装 cvzong  和 mediape
from cvzone.HandTrackingModule import HandDetector

from opencv_test.basic.visul_cal_button import Button

# 详细介绍
# https://blog.csdn.net/dgvv4/article/details/122082894
def calculate():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)  # set：设置窗口大小
    cap.set(4, 720)
    button_values = [['7', '8', '9', '*'],  # 按键符号
                     ['4', '5', '6', '-'],
                     ['1', '2', '3', '+'],
                     ['0', '/', '.', '=']]
    button_list = []
    for x in range(4):
        for y in range(4):
            x_pos = x * 100 + 800
            y_pos = y * 100 + 150
            button = Button((x_pos, y_pos), 100, 100, button_values[y][x])
            button_list.append(button)
    detector = HandDetector(maxHands=1, detectionCon=0.8)  # maxHands：手数量 detectionCon：像手概率
    my_equation = ''
    delay_counter = 0  # 延迟计算
    while True:
        flag, img = cap.read()
        img = cv2.flip(img, 1)  # 摄像头显示的画面和真实画面相反 flip：翻转
        # 检测手, 注意一定要在还没有绘制button之前去检测手
        hands, img = detector.findHands(img, flipType=False)
        if flag:
            for button in button_list:
                button.draw(img)
            cv2.rectangle(img, (800, 70), (800 + 400, 70 + 100), (225, 225, 225), -1)  # 创建显示结果的窗口
            cv2.rectangle(img, (800, 70), (800 + 400, 70 + 100), (50, 50, 50), 3)
            #         print(hands)
            if hands:
                lmlist = hands[0]['lmList']  # 取出食指和中值的点, 计算两者的距离
                length, _, img = detector.findDistance(lmlist[8], lmlist[12], img)
                #             print(length, _, img)
                x, y = lmlist[8]  # 取出手指的坐标
                # 根据食指和中指之间的距离, 如果小于50认为是进行点击操作
                if length < 50 and delay_counter == 0:
                    for i, button in enumerate(button_list):  # enumerate：枚举
                        if button.check_click(x, y):
                            # 说明是一个正确点击. 应该要把点中的数字显示在窗口上
                            my_value = button_values[int(i % 4)][int(i / 4)]
                            if my_value == '=':  # 如果是'='说明要计算
                                try:
                                    my_equation = str(eval(my_equation))  # eval：将字符串相代码一样执行
                                except Exception:
                                    my_equation = ''  # 非法的数学公式 重新输入
                            else:
                                my_equation += my_value  # 字符串的拼接
                            #                             time.sleep(0.1)  # sleep并不能完全解决重复点击的问题
                            delay_counter = 1
            if delay_counter != 0:  # 重置delay_counter 避免重复点击
                delay_counter += 1
                if delay_counter > 10:
                    delay_counter = 0
            cv2.putText(img, my_equation, (810, 130), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
            cv2.imshow('img', img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('c'):
                my_equation = ''  # 清空输出框
        else:
            print('摄像头打开失败')
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    calculate()
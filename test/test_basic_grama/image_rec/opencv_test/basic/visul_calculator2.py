import cv2
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import time
# 详细介绍
# https://blog.csdn.net/dgvv4/article/details/122082894

# 创建按键类
class Button:
    # 初始化，传入pos按键位置，每个矩形框的宽高，矩形框上的数字value
    def __init__(self, pos, width, height, value):
        # 初始化在while循环之前完成
        self.pos = pos
        self.width = width
        self.height = height
        self.value = value

    # 绘图方法在while循环之后完成
    def draw(self, img):

        # 绘制计算器轮廓,img画板,起点坐标,终点坐标,颜色填充
        cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),
                      (225, 225, 225), cv2.FILLED)

        # 给计算器添加边框
        cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),
                      (50, 50, 50), 3)

        # 按键添加文本,img画板,文本内容,坐标,字体,字体大小,字体颜色,线条宽度
        cv2.putText(img, self.value, (self.pos[0] + 30, self.pos[1] + 70),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (50, 50, 50), 2)

    # 点击按钮
    def checkClick(self, x, y):  # 传入食指尖坐标

        # 检查食指x坐标在哪一个按钮框内，x1 < x < x1 + width ，控制一列
        # 检查食指y坐标在哪一个按钮框内，y1 < y < y1 + height ，控制一行
        if self.pos[0] < x < self.pos[0] + self.width and \
                self.pos[1] < y < self.pos[1] + self.height:  # '\'用来换行

            # 如果点击按钮就改变按钮颜色
            cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),
                          (0, 255, 0), cv2.FILLED)

            # 边框还是原来的不变
            cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),
                          (50, 50, 50), 3)

            # 按键文本变颜色，面积变化
            cv2.putText(img, self.value, (self.pos[0] + 30, self.pos[1] + 70),
                        cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 5)

            # 如果成功点击按钮就返回True
            return True

        else:
            return False


# （1）捕获摄像头
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # 显示框的宽1280
cap.set(4, 720)  # 显示框的高720

pTime = 0  # 设置第一帧开始处理的起始时间

# ==1== 手部检测方法，置信度为0.8，最多检测一只手
detector = HandDetector(detectionCon=0.8, maxHands=1)

# ==2== 创建计算器按键
# 创建按钮内容列表
buttonListvalues = [['7', '8', '9', '*'],
                    ['4', '5', '6', '-'],
                    ['1', '2', '3', '+'],
                    ['0', '/', '.', '=']]

buttonList = []  # 存放每个按键的信息

# 创建4*4个按键
for x in range(4):  # 四列
    for y in range(4):  # 四行
        xpos = x * 100 + 800  # 得到四块宽为100的矩形的起点x坐标，从x=800开始
        ypos = y * 100 + 150  # 起点y坐标

        # 传入起点坐标及宽高
        button1 = Button((xpos, ypos), 100, 100, buttonListvalues[y][x])
        buttonList.append(button1)  # 将确定坐标的矩形框信息存入列表中

# ==3== 初始化结果显示框
myEquation = ''
# eval('5'+'5') ==> 10，eval()函数将数字字符串转换成数字计算
delayCounter = 0  # 添加计数器，一次点击触发一次按钮，避免重复

# （2）处理每一帧图像
while True:

    # 接收图片是否导入成功、帧图像
    success, img = cap.read()

    # 翻转图像，保证摄像机画面和人的动作是镜像
    img = cv2.flip(img, flipCode=1)  # 0竖直翻转，1水平翻转

    # （3）检测手部关键点，返回所有绘制后的图像
    hands, img = detector.findHands(img, flipType=False)

    # （4）绘制计算器
    # 绘制计算器显示结果的部分，四个按键的宽合起来是400
    cv2.rectangle(img, (800, 50), (800 + 400, 70 + 100), (225, 225, 225), cv2.FILLED)

    # 结果框轮廓
    cv2.rectangle(img, (800, 50), (800 + 400, 70 + 100), (50, 50, 50), 3)

    # 遍历列表，调用类中的draw方法，绘制每个按键
    for button in buttonList:
        button.draw(img)

    # （5）检测手按了哪个键
    if hands:  # 如果手部关键点返回的列表不为空，证明检测到了手

        # 0代表第一只手，由于我们设置了只检测一只手，所以0就代表检测到的那只
        lmlist = hands[0]['lmList']

        # 获取食指和中指的指尖距离并绘制连线
        # 返回指尖连线长度，线条信息，绘制后的图像
        length, _, img = detector.findDistance(lmlist[8], lmlist[12], img)
        # print(length)

        x, y = lmlist[8]  # 获取食指坐标
        # 如果指尖距离小于50，找到按下了哪个键
        if length < 50:
            for i, button in enumerate(buttonList):  # 遍历所有按键，找到食指尖在哪个按键内

                # 点击按键，按键颜色面积发生变化，返回True。并且延时器为0才能运行
                if button.checkClick(x, y) and delayCounter == 0:

                    # （6）数值计算
                    # 找到点击的按钮的编号i，i是0-15，
                    # 如"4"，索引为4，位置[1][0]，等同于[i%4][i//4]
                    # print(buttonListvalues[i%4][i//4])
                    myValue = buttonListvalues[i % 4][i // 4]

                    # 如果点的是'='号
                    if myValue == '=':
                        # eval()使字符串数字和符号直接做计算, eval('5 * 6 - 2')
                        myEquation = str(eval(myEquation))  # eval返回一个数值

                    else:
                        # 第一次点击"5"，第二次点击"6"，需要显示的是56
                        myEquation += myValue  # 字符串直接相加

                    # 避免重复，方法一，不推荐：
                    # time.sleep(0.2)

                    delayCounter = 1  # 启动计数器，一次运行点击了一个键

    # （7）避免点一次出现多个相同数，方法二：
    # 点击一个按钮之后，delayCounter=1，20帧后才能点击下一个
    if delayCounter != 0:
        delayCounter += 1  # 延迟一帧
        if delayCounter > 50:  # 10帧过去了才能再点击
            delayCounter = 0

    # （8）绘制显示的计算表达式
    cv2.putText(img, myEquation, (800 + 10, 100 + 20), cv2.FONT_HERSHEY_PLAIN,
                3, (50, 50, 50), 3)

    # 查看FPS
    cTime = time.time()  # 处理完一帧图像的时间
    fps = 1 / (cTime - pTime)
    pTime = cTime  # 重置起始时间

    # 在视频上显示fps信息，先转换成整数再变成字符串形式，文本显示坐标，文本字体，文本大小
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 显示图像，输入窗口名及图像数据
    cv2.imshow('image', img)

    # 每帧滞留时间
    key = cv2.waitKey(1)

    # 清空计算器框
    if key == ord('c'):
        myEquation = ''

    # 退出显示
    if key & 0xFF == 27:  # 每帧滞留20毫秒后消失，ESC键退出
        break

# 释放视频资源
cap.release()
cv2.destroyAllWindows()
import cv2


class Button:
    def __init__(self, pos, width, height, value):
        self.pos = pos
        self.width = width
        self.height = height
        self.value = value
    def draw(self, img):
        # 绘制一个计算器的小格子 先画一个实心的灰色矩形
        cv2.rectangle(img, (self.pos[0], self.pos[1]), (self.pos[0] + self.width, self.pos[1] + self.height), (225, 225, 225), -1)
        # 再画矩形的边框
        cv2.rectangle(img, (self.pos[0], self.pos[1]), (self.pos[0] + self.width, self.pos[1] + self.height), (0, 0, 0), 3)
        cv2.putText(img, self.value, (self.pos[0] + 30, self.pos[1] + 70), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 50), 2)
    def check_click(self, x, y,img): # 检测点击点是否小格子内
        if self.pos[0] < x < self.pos[0] +self.width and self.pos[1] < y < self.pos[1] + self.height:
            cv2.rectangle(img, (self.pos[0] + 3, self.pos[1] + 3),
                         (self.pos[0] + self.width -3, self.pos[1]+self.height -3),
                        (255, 255, 255), -1) # 点击中的格子做放大处理
            cv2.putText(img, self.value, (self.pos[0]+25, self.pos[1] + 80), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 5)
            return True
        else:
            return False

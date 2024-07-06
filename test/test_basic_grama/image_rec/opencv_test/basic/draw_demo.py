import cv2
import numpy as np
from matplotlib import pyplot as plt


def draw_mark():
    img = np.ones((600, 800, 3), np.uint8) * 205
    for i in range(7):  # 7 种标记
        cx = 100 * (i + 1)
        color = (0, 0, 255)
        cv2.drawMarker(img, (cx, 100), color, markerType=i, markerSize=10)
        cv2.drawMarker(img, (cx, 200), color, markerType=i, markerSize=20)
        cv2.drawMarker(img, (cx, 300), color, markerType=i, markerSize=30)
        cv2.drawMarker(img, (cx, 400), color, markerType=i, markerSize=40)
        cv2.drawMarker(img, (cx, 500), color, markerType=i, markerSize=50)

    plt.figure(figsize=(9, 4))
    plt.subplot(121), plt.title("Draw marker"), plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(122), plt.title("Draw marker"), plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show() 


def draw_rec_demo():
    height, width, channels = 600, 400, 3
    img = np.ones((height, width, channels), np.uint8) * 192  # 创建黑色图像 RGB=0

    # 围绕矩形中心旋转
    x, y, w, h = (100, 200, 200, 100)  # 左上角坐标 (x,y), 宽度 w，高度 h
    cx, cy = x + w // 2, y + h // 2  # 矩形中心
    img1 = img.copy()
    cv2.circle(img1, (cx, cy), 4, (0, 0, 255), -1)  # 旋转中心
    angle = [15, 30, 45, 60, 75, 90]  # 旋转角度，顺时针方向
    for i in range(len(angle)):
        ang = angle[i] * np.pi / 180
        x1 = int(cx + (w / 2) * np.cos(ang) - (h / 2) * np.sin(ang))
        y1 = int(cy + (w / 2) * np.sin(ang) + (h / 2) * np.cos(ang))
        x2 = int(cx + (w / 2) * np.cos(ang) + (h / 2) * np.sin(ang))
        y2 = int(cy + (w / 2) * np.sin(ang) - (h / 2) * np.cos(ang))
        x3 = int(cx - (w / 2) * np.cos(ang) + (h / 2) * np.sin(ang))
        y3 = int(cy - (w / 2) * np.sin(ang) - (h / 2) * np.cos(ang))
        x4 = int(cx - (w / 2) * np.cos(ang) - (h / 2) * np.sin(ang))
        y4 = int(cy - (w / 2) * np.sin(ang) + (h / 2) * np.cos(ang))
        color = (30 * i, 0, 255 - 30 * i)
        cv2.line(img1, (x1, y1), (x2, y2), color)
        cv2.line(img1, (x2, y2), (x3, y3), color)
        cv2.line(img1, (x3, y3), (x4, y4), color)
        cv2.line(img1, (x4, y4), (x1, y1), color)

    # 围绕矩形左上顶点旋转
    x, y, w, h = (200, 200, 200, 100)  # 左上角坐标 (x,y), 宽度 w，高度 h
    img2 = img.copy()
    cv2.circle(img2, (x, y), 4, (0, 0, 255), -1)  # 旋转中心
    angle = [15, 30, 45, 60, 75, 90, 120, 150, 180, 225]  # 旋转角度，顺时针方向
    for i in range(len(angle)):
        ang = angle[i] * np.pi / 180
        x1, y1 = x, y
        x2 = int(x + w * np.cos(ang))
        y2 = int(y + w * np.sin(ang))
        x3 = int(x + w * np.cos(ang) - h * np.sin(ang))
        y3 = int(y + w * np.sin(ang) + h * np.cos(ang))
        x4 = int(x - h * np.sin(ang))
        y4 = int(y + h * np.cos(ang))
        color = (30 * i, 0, 255 - 30 * i)
        cv2.line(img2, (x1, y1), (x2, y2), color)
        cv2.line(img2, (x2, y2), (x3, y3), color)
        cv2.line(img2, (x3, y3), (x4, y4), color)
        cv2.line(img2, (x4, y4), (x1, y1), color)

    plt.figure(figsize=(9, 6))
    plt.subplot(121), plt.title("img1"), plt.axis('off')
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.subplot(122), plt.title("img2"), plt.axis('off')
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.show() 

if __name__ == '__main__':
    draw_rec_demo()
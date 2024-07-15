import sys
import time

import pygame

def init():

    # 初始化展示模块
    pygame.display.init()
    # 设置窗口大小
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    size = (SCREEN_WIDTH, SCREEN_HEIGHT)
    # 初始化窗口
    window = pygame.display.set_mode(size)
    # 设置窗口标题
    pygame.display.set_caption('Tank Battle')

    blue = 255
    green = 0
    red = 255
    BACKGROUND_COLOR = pygame.Color(blue, green, red)
    window.fill(BACKGROUND_COLOR)


    while 1:
        # 获取键盘事件
        getWindowEvent()

        # 更新窗口
        pygame.display.update()


def getWindowEvent():
    for event in pygame.event.get():
        # 点击窗口右上角的关闭触发的事件
        if event.type == pygame.QUIT:
            sys.exit()
        # 鼠标按下事件
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            print('鼠标按下, 鼠标位置({x}, {y})'.format(x = x, y = y))
        # 鼠标抬起事件
        elif event.type == pygame.MOUSEBUTTONUP:
            print('鼠标抬起')
        # 键盘按键按下事件
        elif event.type == pygame.KEYDOWN:
            print('键盘按键按下', event.key)
            # 具体键盘事件触发
            if event.key == pygame.K_j:
                print('按下键盘 j 键')
        # 键盘按键抬起事件
        elif event.type == pygame.KEYUP:
            print('键盘按键抬起')


if __name__ == '__main__':
    init()
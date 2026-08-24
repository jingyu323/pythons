import sys

import pygame
from Constants import *

class MainGame:
    window = None

    def __init__(self):
        # 初始化展示模块
        pygame.display.init()

        SCREEN_SIZE = (SCREEN_WIDTH, SCREEN_HEIGHT)
        # 初始化窗口
        MainGame.window = pygame.display.set_mode(SCREEN_SIZE)
        # 设置窗口标题
        pygame.display.set_caption('魂斗罗角色')
        # 是否结束游戏
        self.isEnd = False
        # 获取按键
        self.keys = pygame.key.get_pressed()
        # 帧率
        self.fps = 60
        self.clock = pygame.time.Clock()

    def getPlayingModeEvent(self):
        # 获取事件列表
        for event in pygame.event.get():
            # 点击窗口关闭按钮
            if event.type == pygame.QUIT:
                self.isEnd = True
                print("close")
            # 键盘按键按下
            elif event.type == pygame.KEYDOWN:
                self.keys = pygame.key.get_pressed()
                print("key down")
            # 键盘按键抬起
            elif event.type == pygame.KEYUP:
                self.keys = pygame.key.get_pressed()
                print("key up")

    def run(self):
        while not self.isEnd:

            # 设置背景颜色
            pygame.display.get_surface().fill((0, 0, 0))
            # 获取窗口中的事件
            self.getPlayingModeEvent()

            # 更新窗口
            pygame.display.update()

            # 设置帧率
            self.clock.tick(self.fps)
            fps = self.clock.get_fps()
            caption = '魂斗罗 - {:.2f}'.format(fps)
            pygame.display.set_caption(caption)


        else:
            sys.exit()



if __name__ == '__main__':
    MainGame().run()


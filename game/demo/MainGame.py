import sys

import pygame

from demo.tank import PlayerTank

SCREEN_WIDTH = 1100
SCREEN_HEIGHT = 600
BACKGROUND_COLOR = pygame.Color(0, 0, 0)
FONT_COLOR = (255, 255, 255)
PLAYER_TANK_POSITION = (325, 550)

class MainGame:
     window = None
     # 玩家坦克
     playerTank = None

     def startGame(self):
         # 初始化展示模块
         pygame.display.init()
         size = (SCREEN_WIDTH, SCREEN_HEIGHT)
         # 初始化窗口
         MainGame.window = pygame.display.set_mode(size)
         # 设置窗口标题
         pygame.display.set_caption('Tank Battle')

         # 初始化我方坦克
         MainGame.playerTank = PlayerTank(PLAYER_TANK_POSITION[0], PLAYER_TANK_POSITION[1], 1, 1)

         while 1:
             MainGame.window.fill(BACKGROUND_COLOR)

             self.getPlayingModeEvent()

             # 显示我方坦克
             MainGame.playerTank.draw(MainGame.window)
             # 我方坦克移动
             if not MainGame.playerTank.stop:
                 MainGame.playerTank.move()

            # 更新窗口
             pygame.display.update()

     def getPlayingModeEvent(self):
         #
         pygame.key.stop_text_input()
         # 获取所有事件
         eventList = pygame.event.get()
         for event in eventList:

             if event.type == pygame.QUIT:
                 sys.exit()

             if event.type == pygame.KEYDOWN:
                 print('键盘按键按下')
                 if event.key == pygame.K_w:
                     MainGame.playerTank.direction = 'UP'
                     MainGame.playerTank.stop = False
                 elif event.key == pygame.K_s:
                     print('s按下')
                     MainGame.playerTank.direction = 'DOWN'
                     MainGame.playerTank.stop = False
                 elif event.key == pygame.K_a:
                     print('a按下')
                     MainGame.playerTank.direction = 'LEFT'
                     MainGame.playerTank.stop = False
                 elif event.key == pygame.K_d:
                     print('d按下')
                     MainGame.playerTank.direction = 'RIGHT'
                     MainGame.playerTank.stop = False
                 elif event.key == pygame.K_j:
                     print('j按下')

             if event.type == pygame.KEYUP:
                 print('键盘按键抬起')
                 if event.key == pygame.K_w:
                     print('w抬起')
                     MainGame.playerTank.stop = True
                 elif event.key == pygame.K_s:
                     MainGame.playerTank.stop = True
                     print('s抬起')
                 elif event.key == pygame.K_a:
                     MainGame.playerTank.stop = True
                     print('a抬起')
                 elif event.key == pygame.K_d:
                     MainGame.playerTank.stop = True
                     print('d抬起')

if __name__ == '__main__':
    MainGame().startGame()


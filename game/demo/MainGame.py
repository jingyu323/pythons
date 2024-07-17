import sys

import pygame

from demo.EnemyTank import EnemyTank
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
     playerBulletList = []
     playerBulletNumber = 3

     enemyTankList = []
     enemyTankTotalCount = 5
     # 用来给玩家展示坦克的数量
     enemyTankCurrentCount = 5

     # 敌人坦克子弹
     enemyTankBulletList = []


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
             MainGame.playerTank.draw(MainGame.window,PLAYER_TANK_POSITION[0], PLAYER_TANK_POSITION[1])
             # 我方坦克移动
             if not MainGame.playerTank.stop:
                 MainGame.playerTank.move()
                 MainGame.playerTank.collideEnemyTank(MainGame.enemyTankList)

                 # 显示我方坦克子弹
             self.drawPlayerBullet(MainGame.playerBulletList)

             # 展示敌方坦克
             self.drawEnemyTank()

             # 展示敌方坦克子弹
             self.drawEnemyBullet()

            # 更新窗口
             pygame.display.update()

     def drawPlayerBullet(self, playerBulletList):
         # 遍历整个子弹列表，如果是没有被销毁的状态，就把子弹显示出来，否则从列表中删除
         for bullet in playerBulletList:
             if not bullet.isDestroy:
                 bullet.draw(MainGame.window)
                 bullet.move()
                 bullet.playerBulletCollideEnemyTank(MainGame.enemyTankList)
             else:
                 playerBulletList.remove(bullet)

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
                     # 判断子弹数量是否超过指定的个数
                     if len(MainGame.playerBulletList) < MainGame.playerBulletNumber:
                         bullet = MainGame.playerTank.shot()
                         MainGame.playerBulletList.append(bullet)

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


     def drawEnemyTank(self):
        if len(MainGame.enemyTankList) == 0:
            # 一次性产生三个，如果剩余坦克数量超过三，那只能产生三个
            n = min(3, MainGame.enemyTankTotalCount)
            # 如果最小是0，就说明敌人坦克没有了，那么就赢了
            if n == 0:
                print('赢了')
                return
            # 没有赢的话，就产生n个坦克
            self.initEnemyTank(n)
            # 总个数减去产生的个数
            MainGame.enemyTankTotalCount -= n

        for tank in MainGame.enemyTankList:
            # 坦克还有生命值
            if tank.life > 0:
                tank.draw(MainGame.window)
                tank.move()
                tank.collidePlayerTank(MainGame.playerTank)
                tank.collideEnemyTank(MainGame.enemyTankList)

                bullet = tank.shot()
                if bullet is not None:
                    MainGame.enemyTankBulletList.append(bullet)
            # 坦克生命值为0，就从列表中剔除
            else:
                MainGame.enemyTankCurrentCount -= 1
                MainGame.enemyTankList.remove(tank)


     def initEnemyTank(self, number):
            y = 0
            position = [0, 425, 850]
            index = 0
            for i in range(number):
                x = position[index]
                enemyTank = EnemyTank(x, y)
                MainGame.enemyTankList.append(enemyTank)
                index += 1

     def drawEnemyBullet(self):
         for bullet in MainGame.enemyTankBulletList:
             if not bullet.isDestroy:
                 bullet.draw(MainGame.window)
                 bullet.move()

                 bullet.enemyBulletCollidePlayerTank(MainGame.playerTank)
             else:
                 bullet.source.bulletCount -= 1
                 MainGame.enemyTankBulletList.remove(bullet)







if __name__ == '__main__':
    MainGame().startGame()


import sys

import pygame

from demo.BrickWall import BrickWall
from demo.EnemyTank import EnemyTank
from demo.Explode import Explode
from demo.Home import Home
from demo.Sound import Sound
from demo.StoneWall import StoneWall
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
     brickWallList = []
     # 石墙
     stoneWallList = []


     # 爆炸列表
     explodeList = []

     # 坦克移动音效
     playerTankMoveSound = Sound('../tank/Sound/player.move.wav').setVolume()

     # 游戏开始音效
     startingSound = Sound('../tank/Sound/intro.wav')
     isDefeated = False
     home = Home(425, 550)


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
         # 播放开始音乐
         MainGame.startingSound.play()
         # 初始化场景
         self.initBrickWall()

         self.initStoneWall()
         while 1:


             MainGame.window.fill(BACKGROUND_COLOR)

             self.getPlayingModeEvent()

             # 显示物体
             self.drawBrickWall(MainGame.brickWallList)
             self.drawStoneWall(MainGame.stoneWallList)
             MainGame.home.draw(MainGame.window)

             # 展示敌方坦克
             self.drawEnemyTank()
             # 显示我方坦克
             MainGame.playerTank.draw(MainGame.window,PLAYER_TANK_POSITION[0], PLAYER_TANK_POSITION[1])
             # 我方坦克移动
             if not MainGame.playerTank.stop:
                 MainGame.playerTank.move()
                 MainGame.playerTank.collideEnemyTank(MainGame.enemyTankList)
                 # 不能撞墙
                 MainGame.playerTank.collideBrickWall(MainGame.brickWallList)
                 MainGame.playerTank.collideStoneWall(MainGame.stoneWallList)
                 MainGame.playerTank.collideHome(MainGame.home)


                 # 显示我方坦克子弹
             self.drawPlayerBullet(MainGame.playerBulletList )
             # 展示敌方坦克子弹
             self.drawEnemyBullet()
             # 展示爆炸效果
             self.drawExplode()
             if self.isDefeated:
                 self.defeated()
                 break

             # 更新窗口
             pygame.display.update()

         # 设置背景颜色

         MainGame.window.fill(BACKGROUND_COLOR)

         # 显示字体
         self.drawText('Defeated', 200, 200, 50, MainGame.window)

     # 更新窗口
         pygame.display.update()

     def drawPlayerBullet(self, playerBulletList):
         # 遍历整个子弹列表，如果是没有被销毁的状态，就把子弹显示出来，否则从列表中删除
         for bullet in playerBulletList:
             if not bullet.isDestroy:
                 bullet.draw(MainGame.window)
                 bullet.move(MainGame.explodeList)
                 bullet.playerBulletCollideEnemyTank(MainGame.enemyTankList,MainGame.explodeList)
                 bullet.bulletCollideBrickWall(MainGame.brickWallList, MainGame.explodeList)
                 bullet.bulletCollideStoneWall(MainGame.stoneWallList, MainGame.explodeList)


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
                 MainGame.playerTankMoveSound.play(-1)
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
                         Sound('../tank/Sound/shoot.wav').play(0)
                         MainGame.playerBulletList.append(bullet)
                 elif event.key == pygame.K_r:
                     print('r按下')
                     # 判断子弹数量是否超过指定的个数
                     MainGame.playerTank.isResurrecting=True

             if event.type == pygame.KEYUP:
                 MainGame.playerTankMoveSound.stop()
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
                tank.move( )
                tank.collidePlayerTank(MainGame.playerTank)
                tank.collideEnemyTank(MainGame.enemyTankList)
                # 不能撞墙
                tank.collideBrickWall(MainGame.brickWallList)
                tank.collideStoneWall(MainGame.stoneWallList)
                tank.collideHome(MainGame.home)
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
                 bullet.move(MainGame.explodeList)
                 bullet.enemyBulletCollidePlayerTank(MainGame.playerTank, MainGame.explodeList)
                 bullet.bulletCollideBrickWall(MainGame.brickWallList, MainGame.explodeList)
                 bullet.bulletCollideStoneWall(MainGame.stoneWallList, MainGame.explodeList)
                 res = bullet.bulletCollidePlayerHome(MainGame.home, MainGame.explodeList)

                 if res :
                     print(res)
                     MainGame.isDefeated = True
                     self.defeated()
             else:
                 bullet.source.bulletCount -= 1
                 MainGame.enemyTankBulletList.remove(bullet)



     def drawExplode(self):
        for e in MainGame.explodeList:
            if e.isDestroy:
                MainGame.explodeList.remove(e)
            else:
                e.draw(MainGame.window)


     def drawBrickWall(self, brickWallList):
        for brickWall in brickWallList:
            if brickWall.isDestroy:
                brickWallList.remove(brickWall)
            else:
                brickWall.draw(MainGame.window)


     def initBrickWall(self):
        for i in range(20):
            MainGame.brickWallList.append(BrickWall(i * 25, 200))


     def initStoneWall(self):
        for i in range(20):
            MainGame.stoneWallList.append(StoneWall(i * 25, 400))


     def drawStoneWall(self, stoneWallList):
        for stoneWall in stoneWallList:
            if stoneWall.isDestroy:
                stoneWallList.remove(stoneWall)
            else:
                stoneWall.draw(MainGame.window)


     def defeated(self):
        # 失败了坦克不能移动了
        MainGame.playerTankMoveSound.stop()
        # 播放失败音乐
        Sound('../tank/Sound/gameOver.wav').play()
        print('游戏结束')
        MainGame.isDefeated = True

     def drawText(self, text, x, y, fontSize, window):
        # 初始化字体
        pygame.font.init()
        font = pygame.font.SysFont('georgia', fontSize)
        # 加载文字并设置颜色
        fontColor = pygame.Color(255, 255, 255)
        fontObject = font.render(text, True, fontColor)
        # 展示文字
        window.blit(fontObject, (x, y))





if __name__ == '__main__':
    MainGame().startGame()


import math
import random

import pygame
from Constants import *
from Bullet import Bullet
from PlayerOne import PlayerOne


class Enemy2(pygame.sprite.Sprite):

    def __init__(self, x, y, direction, currentTime):
        pygame.sprite.Sprite.__init__(self)
        self.r = 0.0
        self.bulletPosition = 0
        self.rightImage = loadImage('../Image/Enemy/Enemy2/right.png')
        self.rightUpImage = loadImage('../Image/Enemy/Enemy2/rightUp.png')
        self.rightDownImage = loadImage('../Image/Enemy/Enemy2/rightDown.png')
        self.leftImage = loadImage('../Image/Enemy/Enemy2/right.png', True)
        self.leftUpImage = loadImage('../Image/Enemy/Enemy2/rightUp.png', True)
        self.leftDownImage = loadImage('../Image/Enemy/Enemy2/rightDown.png', True)
        self.type = 2
        if direction == Direction.RIGHT:
            self.image = self.rightImage
        else:
            self.image = self.leftImage

        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.center = self.rect.x + self.rect.width / 2, self.rect.y + self.rect.height / 2
        self.isDestroy = False
        self.isFiring = False
        self.life = 1
        self.lastTime = currentTime
        self.n = 0
        # 计算时间
        self.t = 0

    def getCenter(self):
        return self.rect.x + self.rect.width / 2, self.rect.y + self.rect.height / 2

    def draw(self, window: pygame.Surface, player: PlayerOne, currentTime):

        self.n += 1
        # 计算时间
        total = self.t * self.n
        total = total + abs(currentTime - self.lastTime)
        self.lastTime = currentTime
        self.t = total * 1.0 / (self.n + 1)

        # 获取玩家中心
        playerCenter = player.getCenter()
        # 获取敌人中心
        center = self.getCenter()
        # 计算距离
        y = playerCenter[1] - center[1]
        x = playerCenter[0] - center[0]
        # 设置存放夹角的变量
        r = 0
        # 当 x = 0时，此时玩家在敌人的正上方，我们不做任何操作
        if x != 0:
            # 如果玩家在敌人的正上方，计算角度
            r = math.atan(y / x) * 180 / math.pi
        # 设置变量，用来记录敌人的姿势，敌人的姿势就是发射子弹时的样子
        self.bulletPosition = 1
        # 根据距离的正负关系判断玩家在敌人的左边还是右边
        if x >= 0:
            if -45 < r < 45:
                self.bulletPosition = 2
                self.image = self.rightImage
            elif r >= 45:
                self.bulletPosition = 3
                self.image = self.rightDownImage
            elif r <= -45:
                self.bulletPosition = 1
                self.image = self.rightUpImage
        else:
            if -45 < r < 45:
                self.bulletPosition = 5
                self.image = self.leftImage
            elif r <= -45:
                self.bulletPosition = 4
                self.image = self.leftDownImage
            elif r >= 45:
                self.bulletPosition = 6
                self.image = self.leftUpImage
        self.r = r
        window.blit(self.image, self.rect)

    def fire(self, enemyBulletList, player):
        i = random.randint(0, 30)
        if i == 5:
            self.isFiring = True
            enemyBulletList.append(Bullet(self, 2, (self.bulletPosition, player, self.t, self.r)))

    def checkPosition(self, x, y):
        if abs(self.rect.x - x) > 2000:
            self.isDestroy = True
        elif abs(self.rect.y - y) > 600:
            self.isDestroy = True



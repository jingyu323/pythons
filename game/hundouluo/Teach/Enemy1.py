import random

import pygame
from Constants import *
from Bullet import Bullet

class Enemy1(pygame.sprite.Sprite):

    def __init__(self, x, y, direction, currentTime):
        pygame.sprite.Sprite.__init__(self)

        self.lastTime = currentTime
        self.fireTime = currentTime
        self.rightImages = [
            loadImage('../Image/Enemy/Enemy1/1.png'),
            loadImage('../Image/Enemy/Enemy1/2.png'),
            loadImage('../Image/Enemy/Enemy1/3.png')
        ]
        self.leftImages = [
            loadImage('../Image/Enemy/Enemy1/1.png', True),
            loadImage('../Image/Enemy/Enemy1/2.png', True),
            loadImage('../Image/Enemy/Enemy1/3.png', True)
        ]
        self.rightFireImage = loadImage('../Image/Enemy/Enemy1/fire.png')
        self.leftFireImage = loadImage('../Image/Enemy/Enemy1/fire.png', True)
        self.fallImage = loadImage('../Image/Enemy/Enemy1/fall.png', True)

        self.rightFireImage = loadImage('../Image/Enemy/Enemy1/fire.png')
        self.leftFireImage = loadImage('../Image/Enemy/Enemy1/fire.png', True)
        self.rightFallImage = loadImage('../Image/Enemy/Enemy1/fall.png')
        self.leftFallImage = loadImage('../Image/Enemy/Enemy1/fall.png', True)

        self.index = 0
        self.direction = direction
        if self.direction == Direction.RIGHT:
            self.image = self.rightImages[self.index]
        else:
            self.image = self.leftImages[self.index]
        self.rect = self.image.get_rect()
        self.isFalling = False
        self.rect.x = x
        self.rect.y = y
        self.speed = 3
        self.isDestroy = False
        self.isFiring = False
        self.life = 1
        self.type = 1

    def move(self, currentTime):
        # 首先判断敌人是否开火，如果是开火状态，就不能移动
        if not self.isFiring:
            # 没有开火，就根据方向移动，这里我设置敌人只能向一个方向移动，不能转身
            if self.direction == Direction.RIGHT:
                self.rect.left += self.speed
            else:
                self.rect.left -= self.speed
        else:
            # 如果此时是开火状态，判断一下上次开火的时间和这次的时间是否相差1000
            # 这个的作用在于让敌人开火的时候站在那里不动，因为敌人移动时是不能开火的
            if currentTime - self.fireTime > 1000:
                # 如果两次开火间隔相差很大，那么就可以让敌人再次开火
                self.isFiring = False
                self.fireTime = currentTime

    def draw(self, currentTime):
        if self.isFiring:
            if self.direction == Direction.RIGHT:
                self.image = self.rightFireImage
            else:
                self.image = self.leftFireImage
        else:
            if currentTime - self.lastTime > 115:
                if self.index < 2:
                    self.index += 1
                else:
                    self.index = 0
                self.lastTime = currentTime
            if self.direction == Direction.RIGHT:
                self.image = self.rightImages[self.index]
            else:
                self.image = self.leftImages[self.index]

    def fire(self, enemyBulletList):
        if not self.isFalling:
            i = random.randint(0, 50)
            if i == 5:
                if not self.isFiring:
                    self.isFiring = True
                    enemyBulletList.append(Bullet(self, True))

    def checkPosition(self, x, y):
        if abs(self.rect.x - x) > 1000:
            self.isDestroy = True
        elif abs(self.rect.y - y) > 600:
            self.isDestroy = True

import pygame
from Constants import *
from Bullet import Bullet
from PlayerOne import PlayerOne


class BossPart(pygame.sprite.Sprite):
    
    def __init__(self, x, y, currentTime, type = 3):
        pygame.sprite.Sprite.__init__(self)
        # 上一次开火时间
        self.lastTime = currentTime
        self.image = pygame.Surface((20, 20)).convert()
        self.image.fill((255, 0, 0))
        
        # 位置
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.type = type
        self.life = 30
        self.isDestroy = False
        
    def draw(self, window) -> None:
        window.blit(self.image, self.rect)
        
        
    def fire(self, currentTime, enemyBulletList):
        if self.type == 3:
            # 判断时间，如果两次时间间隔大于800，就开火一次
            if currentTime - self.lastTime > 800:
                enemyBulletList.append(Bullet(self, 3))
                self.lastTime = currentTime
    
    
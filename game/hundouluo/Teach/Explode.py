import pygame.image
from Constants import *

class Explode:
    def __init__(self, object, variety = ExplodeVariety.CIRCLE, isUseTime = False):
        # 获取爆炸对象的位置
        self.rect = object.rect
        if variety == ExplodeVariety.CIRCLE:
            self.images = [
                loadImage('../Image/Explode/circleExplode1.png'),
                loadImage('../Image/Explode/circleExplode1.png'),
                loadImage('../Image/Explode/circleExplode1.png'),
                loadImage('../Image/Explode/circleExplode1.png'),
                loadImage('../Image/Explode/circleExplode2.png'),
                loadImage('../Image/Explode/circleExplode2.png'),
                loadImage('../Image/Explode/circleExplode2.png'),
                loadImage('../Image/Explode/circleExplode2.png'),
                loadImage('../Image/Explode/circleExplode3.png'),
                loadImage('../Image/Explode/circleExplode3.png'),
                loadImage('../Image/Explode/circleExplode3.png'),
                loadImage('../Image/Explode/circleExplode3.png'),
            ]
        elif variety == ExplodeVariety.BRIDGE:
            self.images = [
                loadImage('../Image/Explode/bridgeExplode1.png'),
                loadImage('../Image/Explode/bridgeExplode2.png'),
                loadImage('../Image/Explode/bridgeExplode3.png'),
            ]
        elif variety == ExplodeVariety.PLAYER1:
            self.images = [
                loadImage('../Image/Player/Player1/Death/death.png'),
                loadImage('../Image/Player/Player1/Death/death.png'),
                loadImage('../Image/Player/Player1/Death/death.png'),
                loadImage('../Image/Player/Player1/Death/death.png'),
                loadImage('../Image/Player/Player1/Death/death.png'),
            ]
        self.index = 0
        self.image = self.images[self.index]
        self.isDestroy = False
        self.isUseTime = isUseTime
        self.lastTime = None

    def draw(self, window, currentTime = None):
        if self.isUseTime:
            if currentTime - self.lastTime > 115:
                # 根据索引获取爆炸对象, 添加到主窗口
                # 让图像加载五次，这里可以换成五张大小不一样的爆炸图片，可以实现让爆炸效果从小变大的效果
                if self.index < len(self.images):
                    self.image = self.images[self.index]
                    self.index += 1
                    window.blit(self.image, self.rect)
                else:
                    self.isDestroy = True
                    self.index = 0
                self.lastTime = currentTime
            else:
                window.blit(self.image, self.rect)
        else:
            # 根据索引获取爆炸对象, 添加到主窗口
            # 让图像加载五次，这里可以换成五张大小不一样的爆炸图片，可以实现让爆炸效果从小变大的效果
            if self.index < len(self.images):
                self.image = self.images[self.index]
                self.index += 1
                window.blit(self.image, self.rect)
            else:
                self.isDestroy = True
                self.index = 0


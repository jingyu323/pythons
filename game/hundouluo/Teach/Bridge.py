import pygame
from Constants import *

class Bridge:

    def __init__(self, x, y, typeInfo):
        if typeInfo == BridgeType.ON:
            self.image = loadImage('../Image/Map/1/Bridge/bridgeOn.png')
        elif typeInfo == BridgeType.BODY:
            self.image = loadImage('../Image/Map/1/Bridge/bridgeBody.png')
        else:
            self.image = loadImage('../Image/Map/1/Bridge/bridgeDown.png')

        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

        self.isDestroy = False

    def draw(self, window: pygame.Surface):
        window.blit(self.image, self.rect)

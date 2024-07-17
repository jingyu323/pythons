import pygame

from demo.ParentObject import ParentObject

class Wall(ParentObject):
    def __init__(self, x, y):
        super().__init__()


        self.rect.left = x
        self.rect.top = y
        self.isDestroy = False

    def draw(self, window):
        window.blit(self.image, self.rect)

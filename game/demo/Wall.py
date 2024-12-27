import pygame

from demo.ParentObject import ParentObject

class Wall(ParentObject):
    def __init__(self):
        super().__init__()



    def draw(self, window):
        window.blit(self.image, self.rect)

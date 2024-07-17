import pygame

from demo.Wall import Wall


class BrickWall(Wall):
    def __init__(self, x, y):
        super().__init__(x,y)
        self.image = pygame.image.load('../tank/Image/Wall/BrickWall.png')
        self.rect = self.image.get_rect()

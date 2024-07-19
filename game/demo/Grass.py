import pygame.image

from demo.Wall import Wall


class Grass(Wall):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.image.load('../tank/Image/Wall/Grass25x25.png')
        self.rect = self.image.get_rect()
        self.rect.left = x
        self.rect.top = y
        self.isDestroy = False



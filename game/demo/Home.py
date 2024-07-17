import pygame

from demo.Wall import Wall
class Home(Wall):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.image.load('../tank/Image/Home/Home.png')
        self.rect = self.image.get_rect()
        self.rect.left = x
        self.rect.top = y
        self.isDestroy = False






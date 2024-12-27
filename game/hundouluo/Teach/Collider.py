import pygame
from Constants import *


class Collider(pygame.sprite.Sprite):

    def __init__(self, x, y, width, height, color = (255, 0, 0)):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.Surface((width, height)).convert()
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

    def draw(self, window, y):
        if y > self.rect.bottom:
            return False
        else:
            window.blit(self.image, self.rect)
            return True

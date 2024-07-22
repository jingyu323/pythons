# ContraMap.py
import os
import pygame

from Config import Constant, Variable


class StageMap(pygame.sprite.Sprite):
    def __init__(self, order):
        pygame.sprite.Sprite.__init__(self)
        self.image_list = []
        self.order = order
        for i in range(1, 7):
            image = pygame.image.load(os.path.join('image', 'map', 'stage' + str(i) + '.png'))
            rect = image.get_rect()
            image = pygame.transform.scale(image, (rect.width * Constant.MAP_SCALE, rect.height * Constant.MAP_SCALE))
            self.image_list.append(image)
        self.image = self.image_list[self.order]
        self.rect = self.image.get_rect()
        self.rect.x = 0
        self.rect.y = 0
        self.speed = 0

    def update(self):
        if self.order == 2:
            print('纵向地图')
        else:
            if Variable.step == 0 and self.rect.x >= -Constant.WIDTH:
                self.rect.x -= 10
            if Variable.step == 1 and self.rect.x > -Constant.WIDTH * 2:
                self.rect.x -= 10
                if self.rect.x == -2400:
                    Variable.step = 2
            if Variable.step == 2:
                self.rect.x -= self.speed


def new_stage():
    if Variable.map_switch:
        stage_map = StageMap(Variable.stage - 1)
        Variable.all_sprites.add(stage_map)
        Variable.map_switch = False
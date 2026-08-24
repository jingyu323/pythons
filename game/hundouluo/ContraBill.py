# ContraBill.py
import os
import pygame

from Config import Constant, Variable


class Bill(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image_dict = {
            'be_hit': [],
            'jump': [],
            'oblique_down': [],
            'oblique_up': [],
            'run': [],
            'shoot': [],
            'stand': [],
            'up': []
        }
        for i in range(6):
            for key in self.image_dict:
                image = pygame.image.load(os.path.join('image', 'bill', str(key), str(key) + str(i + 1) + '.png'))
                rect = image.get_rect()
                image_scale = pygame.transform.scale(
                    image, (rect.width * Constant.MAP_SCALE, rect.height * Constant.MAP_SCALE))
                self.image_dict[key].append(image_scale)

            self.image_order = 0
            self.image_type = 'stand'

            self.image = self.image_dict[self.image_type][self.image_order]
            self.rect = self.image.get_rect()
            self.rect.x = 100
            self.rect.y = 250
            self.direction = 'right'

    def update(self):
        self.image = self.image_dict[self.image_type][self.image_order]
        if self.direction == 'left':
            self.image = pygame.transform.flip(self.image, True, False)

        key_pressed = pygame.key.get_pressed()
        if key_pressed[pygame.K_a]:
            self.direction = 'left'
            self.image_type = 'run'
            self.rect.x -= 2
        elif key_pressed[pygame.K_d]:
            self.direction = 'right'
            self.image_type = 'run'
            self.rect.x += 2


def new_player():
    bill = Bill()
    if Variable.step == 2 and Variable.player_switch:
        Variable.all_sprites.add(bill)
        Variable.player_switch = False
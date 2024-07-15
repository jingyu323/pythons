# Contra.py
import sys
import pygame

import ContraBill
import ContraMap
from Config import Constant, Variable


def control():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            Variable.game_start = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                Variable.step = 1


class Main:
    def __init__(self):
        pygame.init()
        self.game_window = pygame.display.set_mode((Constant.WIDTH, Constant.HEIGHT))
        self.clock = pygame.time.Clock()

    def game_loop(self):
        while Variable.game_start:
            control()
            if Variable.stage == 1:
                ContraMap.new_stage()
                ContraBill.new_player()

            if Variable.stage == 2:
                pass

            if Variable.stage == 3:
                pass

            Variable.all_sprites.draw(self.game_window)
            Variable.all_sprites.update()
            print(Variable.all_sprites)

            self.clock.tick(Constant.FPS)
            pygame.display.set_caption(f'魂斗罗  1.0    {self.clock.get_fps():.2f}')
            pygame.display.update()

        pygame.quit()
        sys.exit()


if __name__ == '__main__':
    main = Main()
    main.game_loop()
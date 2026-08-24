# Config.py
import pygame


class Constant:
    WIDTH = 1200
    HEIGHT = 720
    FPS = 60

    MAP_SCALE = 3


class Variable:
    all_sprites = pygame.sprite.Group()

    game_start = True
    map_switch = True
    player_switch = True

    stage = 1
    step = 0
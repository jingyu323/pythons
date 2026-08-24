from enum import Enum
import pygame

class State(Enum):
    STAND = 1
    WALK = 2
    JUMP = 3
    FALL = 4
    SQUAT = 5

class Direction(Enum):
    RIGHT = 1
    LEFT = 2

class ExplodeVariety(Enum):
    CIRCLE = 1
    BRIDGE = 2
    PLAYER1 = 3

# 玩家放缩倍数
PLAYER_SCALE = 1.9

def loadImage(filename, hReverse = False):
    image = pygame.image.load(filename)
    if hReverse:
        image = pygame.transform.flip(image, True, False)
    rect = image.get_rect()
    image = pygame.transform.scale(
        image,
        (int(rect.width * PLAYER_SCALE), int(rect.height * PLAYER_SCALE))
    )
    image = image.convert_alpha()
    return image

# 设置窗口大小
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 800
GROUND_HEIGHT = 295

# 玩家x方向速度
PLAYER_X_SPEED = 4
# 设置玩家子弹上限
PLAYER_BULLET_NUMBER = 15

# 地图放缩倍数
MAP_SCALE = 2.5

# 陆地的厚度
LAND_THICKNESS = 1
# 一块地的长度
LAND_LENGTH = 32

POSITION_1 = 233

RATIO = 1.3571428571428572

#中心偏移量
y0 = 0

class BridgeType(Enum):
    ON = 1
    BODY = 2
    DOWN = 3
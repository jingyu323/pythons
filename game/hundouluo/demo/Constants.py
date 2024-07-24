from enum import Enum

# 玩家的四种状态
class State(Enum):
    STAND = 1
    WALK = 2
    JUMP = 3
    FALL = 4

# 玩家的方向
class Direction(Enum):
    RIGHT = 1
    LEFT = 2


# 设置窗口大小
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 800
GROUND_HEIGHT = 63

# 玩家x方向速度
PLAYER_X_SPEED = 3

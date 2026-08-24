import pygame
import random
import os
import sys

# 初始化 pygame
pygame.init()

# 屏幕大小
WIDTH, HEIGHT = 600, 400
CELL_SIZE = 20

# 定义颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
TURQUOISE = (187, 255, 255)
CYAN = (0, 229, 238)
CYANA = (0, 255, 255)

# 创建屏幕
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('贪吃蛇')

# 初始化时钟
clock = pygame.time.Clock()

# 最高分保存地址
save_path = 'score.txt'


# 按钮类
class Button:
    def __init__(self, text, x, y, width, height, font_size, font_color):
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)
        self.font = pygame.font.Font(None, font_size)
        self.text_surf = self.font.render(self.text, True, font_color)
        self.text_rect = self.text_surf.get_rect(center=self.rect.center)

    def draw(self, screenn, bg_color):
        pygame.draw.rect(screenn, bg_color, self.rect)
        screen.blit(self.text_surf, self.text_rect)

    def is_clicked(self, pos):
        if self.rect.collidepoint(pos):
            return True


# 创建按钮
play_again_button = Button('Play Again', WIDTH // 2 - 100, HEIGHT // 2 + 70, 200, 50, 30, WHITE)
game_over_button = Button('Exit', WIDTH // 2 - 100, HEIGHT // 2 + 20, 200, 50, 30, WHITE)
easy_button = Button('Easy', WIDTH // 2 - 100, HEIGHT // 2 - 50, 200, 30, 30, WHITE)
common_button = Button('Common', WIDTH // 2 - 100, HEIGHT // 2 - 20, 200, 30, 30, WHITE)
difficult_button = Button('Diffcult', WIDTH // 2 - 100, HEIGHT // 2 + 10, 200, 30, 30, WHITE)
play_button = Button('play', WIDTH // 2 - 100, HEIGHT // 2 + 50, 200, 50, 30, WHITE)


def draw_snake(snake_body):
    """绘制蛇的身体"""
    for segment in snake_body:
        pygame.draw.rect(screen, GREEN, pygame.Rect(segment[0], segment[1], CELL_SIZE, CELL_SIZE))


def show_message(text, color, size, position):
    """显示信息"""
    font = pygame.font.Font(None, size)
    message = font.render(text, True, color)
    screen.blit(message, position)


def game():
    # 初始化本局得分和最高纪录
    score = 0
    maxscore = 0

    # 检查score文件是否存在
    if not os.path.exists(save_path):
        with open(save_path, 'w'):
            pass
    else:
        try:
            with open(save_path, 'r') as file:
                maxscore = int(file.read().strip())
        except ValueError:
            maxscore = 0
        except FileNotFoundError:
            maxscore = 0

    # 选择游戏难度界面
    rundevel = True
    nodickdevel = True
    SCR = 3
    while rundevel:
        screen.fill(BLACK)
        show_message("Greedy Snake", RED, 50, (WIDTH // 2 - 110, HEIGHT // 2 - 130))
        easy_button.draw(screen, TURQUOISE)
        common_button.draw(screen, TURQUOISE)
        difficult_button.draw(screen, TURQUOISE)
        play_button.draw(screen, GREEN)
        pygame.display.flip()
        if nodickdevel == True:
            if easy_button.is_clicked(pygame.mouse.get_pos()):
                easy_button.draw(screen, CYAN)
                common_button.draw(screen, TURQUOISE)
                difficult_button.draw(screen, TURQUOISE)
                pygame.display.flip()
            elif common_button.is_clicked(pygame.mouse.get_pos()):
                easy_button.draw(screen, TURQUOISE)
                common_button.draw(screen, CYAN)
                difficult_button.draw(screen, TURQUOISE)
                pygame.display.flip()
            elif difficult_button.is_clicked(pygame.mouse.get_pos()):
                easy_button.draw(screen, TURQUOISE)
                common_button.draw(screen, TURQUOISE)
                difficult_button.draw(screen, CYAN)
                pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if easy_button.is_clicked(event.pos):
                    SCR = 0
                    movespeed = 4
                    nodickdevel = False
                elif common_button.is_clicked(event.pos):
                    SCR = 1
                    movespeed = 7
                    nodickdevel = False
                elif difficult_button.is_clicked(event.pos):
                    SCR = 2
                    movespeed = 10
                    nodickdevel = False
                elif play_button.is_clicked(event.pos) and (SCR == 0 or SCR == 1 or SCR == 2):
                    rundevel = False
        if SCR == 0:
            easy_button.draw(screen, CYANA)
            common_button.draw(screen, TURQUOISE)
            difficult_button.draw(screen, TURQUOISE)
            pygame.display.flip()
        elif SCR == 1:
            easy_button.draw(screen, TURQUOISE)
            common_button.draw(screen, CYANA)
            difficult_button.draw(screen, TURQUOISE)
            pygame.display.flip()
        elif SCR == 2:
            easy_button.draw(screen, TURQUOISE)
            common_button.draw(screen, TURQUOISE)
            difficult_button.draw(screen, CYANA)
            pygame.display.flip()
        clock.tick(30)

    # 初始化蛇
    snake_pos = [100, 40]  # 蛇头初始位置
    snake_body = [[100, 40], [80, 40], [60, 40]]  # 蛇身体
    direction = 'RIGHT'  # 蛇头初始方向
    change_to = direction

    # 初始化食物
    food_pos = [random.randrange(1, WIDTH // CELL_SIZE) * CELL_SIZE,
                random.randrange(1, HEIGHT // CELL_SIZE) * CELL_SIZE]
    food_spawn = True

    # 游戏进行界面
    running = True
    while running:
        # 监听按键事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and direction != 'DOWN':
                    change_to = 'UP'
                elif event.key == pygame.K_DOWN and direction != 'UP':
                    change_to = 'DOWN'
                elif event.key == pygame.K_LEFT and direction != 'RIGHT':
                    change_to = 'LEFT'
                elif event.key == pygame.K_RIGHT and direction != 'LEFT':
                    change_to = 'RIGHT'

        direction = change_to

        # 更新蛇头位置
        if direction == 'UP':
            snake_pos[1] -= CELL_SIZE
        elif direction == 'DOWN':
            snake_pos[1] += CELL_SIZE
        elif direction == 'LEFT':
            snake_pos[0] -= CELL_SIZE
        elif direction == 'RIGHT':
            snake_pos[0] += CELL_SIZE

        # 增加蛇头
        snake_body.insert(0, list(snake_pos))

        # 检测是否吃到食物
        if snake_pos == food_pos:
            score += 10  # 更新分数
            food_spawn = False
        else:
            snake_body.pop()

        # 刷新食物位置
        if not food_spawn:
            food_pos = [random.randrange(1, WIDTH // CELL_SIZE) * CELL_SIZE,
                        random.randrange(1, HEIGHT // CELL_SIZE) * CELL_SIZE]
            food_spawn = True

        # 游戏结束条件
        if (
                snake_pos[0] < 0 or snake_pos[0] >= WIDTH or
                snake_pos[1] < 0 or snake_pos[1] >= HEIGHT
        ):
            running = False

        # 检测蛇是否碰到自己
        for block in snake_body[1:]:
            if snake_pos == block:
                running = False

        # 绘制屏幕
        screen.fill(BLACK)
        draw_snake(snake_body)
        pygame.draw.rect(screen, RED, pygame.Rect(food_pos[0], food_pos[1], CELL_SIZE, CELL_SIZE))
        show_message(f'Score: {score}', WHITE, 20, (10, 10))  # 确保每次循环都更新分数显示
        show_message(f'Max_score: {maxscore}', WHITE, 20, (490, 10))
        pygame.display.update()
        clock.tick(movespeed)

    # 游戏结束界面
    run = True
    while run:
        # 游戏结束信息
        if score > maxscore:
            with open(save_path, 'w') as file:
                file.write(str(score))
        screen.fill(BLACK)
        show_message("Game Over!", RED, 50, (WIDTH // 2 - 110, HEIGHT // 2 - 130))
        show_message(f"Your Score: {score}", WHITE, 30, (WIDTH // 2 - 100, HEIGHT // 2 - 80))
        show_message(f"History MaxScore: {maxscore}", WHITE, 30, (WIDTH // 2 - 100, HEIGHT // 2 - 30))
        play_again_button.draw(screen, GREEN)
        game_over_button.draw(screen, RED)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if play_again_button.is_clicked(event.pos):
                    game()
                    run = False
                if game_over_button.is_clicked(event.pos):
                    pygame.quit()
                    run = False


if __name__ == "__main__":
    game()
import pygame

# 初始化展示模块
pygame.display.init()

# 设置窗口大小
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
size = (SCREEN_WIDTH, SCREEN_HEIGHT)
# 初始化窗口
window = pygame.display.set_mode(size)
# 设置窗口标题
pygame.display.set_caption('Contra')

bg = pygame.image.load('../Image/Map/1/Background/First(Bridge).png')
rect = bg.get_rect()
bg = pygame.transform.scale(bg, (int(rect.width * 2.5), int(rect.height * 2.5)))
window.blit(bg, (-700, 0))

clock = pygame.time.Clock()
fps = 60

while 1:
    # 设置帧率
    clock.tick(fps)
    r = clock.get_fps()
    caption = 'Contra - {:.2f}'.format(r)
    pygame.display.set_caption(caption)

    # 更新窗口
    pygame.display.update()




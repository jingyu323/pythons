import pygame


class ParentObject(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()

class PlayerTank(ParentObject):
    def __init__(self, x, y, order, amour):
        """

                :param x: 坦克横坐标
                :param y: 坦克纵坐标
                :param order: 玩家坦克序号，1表示一号玩家，2表示二号玩家
                :param amour: 坦克初始护甲
        """
        super().__init__()
        self.images = []
        if order == 1:
            self.images.append({
                'UP': pygame.image.load('../tank/Image/Player1/45x45/UP1.png'),
                'DOWN': pygame.image.load('../tank/Image/Player1/45x45/DOWN1.png'),
                'LEFT': pygame.image.load('../tank/Image/Player1/45x45/LEFT1.png'),
                'RIGHT': pygame.image.load('../tank/Image/Player1/45x45/RIGHT1.png')
            })
            self.images.append({
                'UP': pygame.image.load('../tank/Image/Player1/45x45/UP2.png'),
                'DOWN': pygame.image.load('../tank/Image/Player1/45x45/DOWN2.png'),
                'LEFT': pygame.image.load('../tank/Image/Player1/45x45/LEFT2.png'),
                'RIGHT': pygame.image.load('../tank/Image/Player1/45x45/RIGHT2.png')
            })
            self.images.append({
                'UP': pygame.image.load('../tank/Image/Player1/45x45/UP3.png'),
                'DOWN': pygame.image.load('../tank/Image/Player1/45x45/DOWN3.png'),
                'LEFT': pygame.image.load('../tank/Image/Player1/45x45/LEFT3.png'),
                'RIGHT': pygame.image.load('../tank/Image/Player1/45x45/RIGHT3.png')
            })
            self.images.append({
                'UP': pygame.image.load('../tank/Image/Player1/45x45/UP4.png'),
                'DOWN': pygame.image.load('../tank/Image/Player1/45x45/DOWN4.png'),
                'LEFT': pygame.image.load('../tank/Image/Player1/45x45/LEFT4.png'),
                'RIGHT': pygame.image.load('../tank/Image/Player1/45x45/RIGHT4.png')
            })
            self.images.append({
                'UP': pygame.image.load('../tank/Image/Player1/45x45/UP5.png'),
                'DOWN': pygame.image.load('../tank/Image/Player1/45x45/DOWN5.png'),
                'LEFT': pygame.image.load('../tank/Image/Player1/45x45/LEFT5.png'),
                'RIGHT': pygame.image.load('../tank/Image/Player1/45x45/RIGHT5.png')
            })
            self.images.append({
                'UP': pygame.image.load('../tank/Image/Player1/45x45/UP6.png'),
                'DOWN': pygame.image.load('../tank/Image/Player1/45x45/DOWN6.png'),
                'LEFT': pygame.image.load('../tank/Image/Player1/45x45/LEFT6.png'),
                'RIGHT': pygame.image.load('../tank/Image/Player1/45x45/RIGHT6.png')
            })

        # 生命
        self.life = 3
        # 装甲
        self.armor = amour

        # 方向
        self.direction = 'UP'

        # 根据护甲选择坦克的样子
        self.image: pygame.Surface = self.images[max(self.armor - 1, 0)][self.direction]
        self.rect = self.image.get_rect()
        self.rect.left = x
        self.rect.top = y

        # 速度
        self.accumulation: float = 0
        self.speed = 2
        # 移动开关
        self.stop = True

        # 等级
        self.level = 1
        # 伤害
        self.damage = 1


    def move(self):
        if self.accumulation >= 1:
            self.accumulation = 0
            if self.direction == 'LEFT':
                if self.rect.left > 0:
                    self.rect.left -= self.speed
            elif self.direction == 'UP':
                if self.rect.top > 0:
                    self.rect.top -= self.speed
            elif self.direction == 'DOWN':
                if self.rect.top < 555:
                    self.rect.top += self.speed
            elif self.direction == 'RIGHT':
                if self.rect.left < 855:
                    self.rect.left += self.speed
        else:
            self.accumulation += 0.20


    def shot(self):
        pass


    def draw(self,window):
        # window传入主窗口
        # 坦克生命中为0，表示已经死亡，不再展示坦克
        if self.life <= 0:
            return
        # 获取展示的对象
        self.image = self.images[max(self.armor - 1, 0)][self.direction]
        window.blit(self.image, self.rect)




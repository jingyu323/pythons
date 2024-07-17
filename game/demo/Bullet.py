
import pygame

from demo.ParentObject import ParentObject


class Bullet(ParentObject):
    def __init__(self, tank):
        super().__init__()
        self.images = {
            'UP': pygame.image.load('../tank/Image/Bullet/Bullet(UP).png'),
            'DOWN': pygame.image.load('../tank/Image/Bullet/Bullet(DOWN).png'),
            'LEFT': pygame.image.load('../tank/Image/Bullet/Bullet(LEFT).png'),
            'RIGHT': pygame.image.load('../tank/Image/Bullet/Bullet(RIGHT).png')
        }

        # 方向
        self.direction = tank.direction
        self.image : pygame.Surface = self.images[self.direction]
        self.rect = self.image.get_rect()
        # 坦克发射子弹的位置
        if self.direction == 'UP':
            self.rect.left = tank.rect.left + 17.5
            self.rect.top = tank.rect.top - 25
        elif self.direction == 'DOWN':
            self.rect.left = tank.rect.left + 17.5
            self.rect.top = tank.rect.top + 25
        elif self.direction == 'LEFT':
            self.rect.left = tank.rect.left - 25
            self.rect.top = tank.rect.top + 17.5
        elif self.direction == 'RIGHT':
            self.rect.left = tank.rect.left + 25
            self.rect.top = tank.rect.top + 17.5

        # 速度
        self.accumulationMax: float = 0
        self.accumulation = 0.25
        self.speed = 10
        # 销毁开关
        self.isDestroy = False
        # 发射源
        self.source = tank
        # 伤害
        self.damage = tank.damage

    def move(self):
        if self.accumulation >= 1:
            self.accumulation = 0
            if self.direction == 'LEFT':
                self.rect.left -= self.speed
            elif self.direction == 'UP':
                self.rect.top -= self.speed
            elif self.direction == 'DOWN':
                self.rect.top += self.speed
            elif self.direction == 'RIGHT':
                self.rect.left += self.speed
            # 检查子弹是否出界
            self.checkBullet()
        else:
            self.accumulation += 0.20

    def draw(self, window):
        window.blit(self.image, self.rect)

    def checkBullet(self):
        toDestroy = False
        # 如果出界，就设置为销毁
        if self.rect.top < 0 or self.rect.top > 600:
            toDestroy = True
        if self.rect.left < 0 or self.rect.right > 900:
            toDestroy = True
        if toDestroy:
            self.isDestroy = True

    def playerBulletCollideEnemyTank(self, enemyTankList):
        # 循环遍历坦克列表，检查是否发生了碰撞
        for tank in enemyTankList:
            if pygame.sprite.collide_rect(tank, self):
                # 坦克被击中就扣减生命值
                tank.loseLife(self.damage)
                # 把子弹设置为销毁状态
                self.isDestroy = True

    def enemyBulletCollidePlayerTank(self, playerTank):
        # 玩家坦克生命值为0，不用检测
        if playerTank.life <= 0:
            return
        # 检测是否发生碰撞
        if pygame.sprite.collide_rect(playerTank, self):
            # 发生碰撞先减少护甲，护甲为0时扣减生命值
            if playerTank.armor > 0:
                playerTank.armor -= self.damage
                playerTank.armor = max(0, playerTank.armor)
            else:
                playerTank.loseLife(self.damage)
                playerTank.life = max(0, playerTank.life)
                if playerTank.life != 0:
                    playerTank.isResurrecting = True
            # 让子弹销毁
            self.isDestroy = True

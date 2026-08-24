import pygame
from Constants import *
from Explode import Explode
from Sound import Sound

class Bullet(pygame.sprite.Sprite):

    def __init__(self, person, enemyType = 0, parameter = None):
        pygame.sprite.Sprite.__init__(self)
        self.images = [
            loadImage('../Image/Bullet/bullet1.png'),
            loadImage('../Image/Bullet/bullet2.png'),
            loadImage('../Image/Bullet/bullet3.png'),
        ]
        self.index = 0
        # 速度
        self.xSpeed = 1
        self.ySpeed = 1
        # 加速度
        self.xAcc = 0
        self.yAcc = 0
        self.rect = pygame.Rect(person.rect)
        self.enemyType = enemyType
        # 类型0表示不是敌人
        if enemyType == 0:
            if person.isInWater:
                self.waterPosition(person)
            else:
                self.landPosition(person)

        # 敌人1
        elif enemyType == 1:

            self.index = 0
            if person.direction == Direction.RIGHT:
                self.rect.x += 27 * PLAYER_SCALE
                self.rect.y += 7 * PLAYER_SCALE
                self.ySpeed = 0
                self.xSpeed = 7
            else:
                self.rect.x += -1 * PLAYER_SCALE
                self.rect.y += 7 * PLAYER_SCALE
                self.ySpeed = 0
                self.xSpeed = -7
        # 敌人2
        elif enemyType == 2:
            self.index = 0
            # 从额外参数中获取敌人的姿势，即子弹的发射位置
            bulletPosition = parameter[0]
            # 获取玩家对象
            player = parameter[1]
            # 获取玩家中心
            playerCenter = player.getCenter()
            # 让人物中心下移
            if player.isDown or player.isSquating:
                # 下蹲、蹲下、在水中时，让人物中心下移动，下移动代表y坐标的值相加
                playerCenter = (playerCenter[0], playerCenter[1] + 8)
            elif player.isInWater:
                playerCenter = (playerCenter[0], playerCenter[1] + 15)
            # 获取子弹移动的时间
            t = parameter[2]
            # t *= 15
            # 获取敌人与玩家连线与水平方向的夹角
            r = parameter[3]
            # 根据子弹的发射位置（敌人的姿势）计算敌人的发射子弹的位置和子弹的速度
            if bulletPosition == 1:
                self.rect.x += 19 * PLAYER_SCALE
                self.rect.y += -1 * PLAYER_SCALE
                # 计算公式，|x0 - x1| / t = v
                self.ySpeed = - abs(self.rect.y - playerCenter[1]) * 1.0 / t
                self.xSpeed = abs(self.rect.x - playerCenter[0]) * 1.0 / t
            elif bulletPosition == 2:
                self.rect.x += 25 * PLAYER_SCALE
                self.rect.y += 10 * PLAYER_SCALE
                # s 表示方向这里可以直接根据r的大小，计算出子弹的速度是减少还是增加
                # 减少表示向负方向移动
                s = -1
                if r > 0:
                    s = 1
                self.ySpeed = s * abs(self.rect.y - playerCenter[1]) * 1.0 / t
                self.xSpeed = abs(self.rect.x - playerCenter[0]) * 1.0 / t
            elif bulletPosition == 3:
                self.rect.x += 25 * PLAYER_SCALE
                self.rect.y += 25 * PLAYER_SCALE
                self.ySpeed = abs(self.rect.y - playerCenter[1]) * 1.0 / t
                self.xSpeed = abs(self.rect.x - playerCenter[0]) * 1.0 / t
            elif bulletPosition == 4:
                self.rect.x += -1 * PLAYER_SCALE
                self.rect.y += 25 * PLAYER_SCALE
                self.ySpeed = abs(self.rect.y - playerCenter[1]) * 1.0 / t
                self.xSpeed = - abs(self.rect.x - playerCenter[0]) * 1.0 / t
            elif bulletPosition == 5:
                self.rect.x += -1 * PLAYER_SCALE
                self.rect.y += 10 * PLAYER_SCALE
                s = 1
                if r > 0:
                    s = -1
                self.ySpeed = s * abs(self.rect.y - playerCenter[1]) * 1.0 / t
                self.xSpeed = - abs(self.rect.x - playerCenter[0]) * 1.0 / t
            elif bulletPosition == 6:
                self.rect.x += -1 * PLAYER_SCALE
                self.rect.y += -1 * PLAYER_SCALE
                self.ySpeed = - abs(self.rect.y - playerCenter[1]) * 1.0 / t
                self.xSpeed = - abs(self.rect.x - playerCenter[0]) * 1.0 / t
            self.xSpeed /= 5
            self.ySpeed /= 5
        elif enemyType == 3:
            self.index = 2
            self.xSpeed = -5
            self.ySpeed = 3
            self.xAcc = 0.2
            self.yAcc = 0.4
        self.image = self.images[self.index]

        # 销毁开关
        self.isDestroy = False

    def landPosition(self, person):
        if person.isStanding:
            if person.direction == Direction.RIGHT:
                if person.isUp:
                    self.rect.x += 10 * PLAYER_SCALE
                    self.rect.y += -1 * PLAYER_SCALE
                    self.ySpeed = -7
                    self.xSpeed = 0
                else:
                    self.rect.x += 24 * PLAYER_SCALE
                    self.rect.y += 11 * PLAYER_SCALE
                    self.ySpeed = 0
                    self.xSpeed = 7
            else:
                if person.isUp:
                    self.rect.x += 10 * PLAYER_SCALE
                    self.rect.y += -1 * PLAYER_SCALE
                    self.ySpeed = -7
                    self.xSpeed = 0
                else:
                    self.rect.y += 11 * PLAYER_SCALE
                    self.ySpeed = 0
                    self.xSpeed = -7

        elif person.isSquating and not person.isWalking:
            if person.direction == Direction.RIGHT:
                self.rect.x += 34 * PLAYER_SCALE
                self.rect.y += 25 * PLAYER_SCALE
                self.ySpeed = 0
                self.xSpeed = 7
            else:
                self.rect.y += 25 * PLAYER_SCALE
                self.ySpeed = 0
                self.xSpeed = -7

        elif person.isWalking:
            if person.direction == Direction.RIGHT:
                if person.isUp:
                    self.rect.x += 20 * PLAYER_SCALE
                    self.rect.y += -1 * PLAYER_SCALE
                    self.ySpeed = -7
                    self.xSpeed = 7
                elif person.isDown:
                    self.rect.x += 21 * PLAYER_SCALE
                    self.rect.y += 20 * PLAYER_SCALE
                    self.ySpeed = 7
                    self.xSpeed = 7
                else:
                    self.rect.x += 24 * PLAYER_SCALE
                    self.rect.y += 11 * PLAYER_SCALE
                    self.ySpeed = 0
                    self.xSpeed = 7
            else:
                if person.isUp:
                    self.rect.x += -3 * PLAYER_SCALE
                    self.rect.y += -1 * PLAYER_SCALE
                    self.ySpeed = -7
                    self.xSpeed = -7
                elif person.isDown:
                    self.rect.x += -3 * PLAYER_SCALE
                    self.rect.y += 20 * PLAYER_SCALE
                    self.ySpeed = 7
                    self.xSpeed = -7
                else:
                    self.rect.y += 11 * PLAYER_SCALE
                    self.ySpeed = 0
                    self.xSpeed = -7

        elif person.isJumping or person.state == State.FALL:
            if person.direction == Direction.RIGHT:
                self.rect.x += 16 * PLAYER_SCALE
                self.rect.y += 8 * PLAYER_SCALE
                self.ySpeed = 0
                self.xSpeed = 7
            else:
                self.rect.x += -2 * PLAYER_SCALE
                self.rect.y += 8 * PLAYER_SCALE
                self.ySpeed = 0
                self.xSpeed = -7

    def waterPosition(self, person):
        if person.isStanding:
            if person.direction == Direction.RIGHT:
                if person.isUp:
                    self.rect.x += 14 * PLAYER_SCALE
                    self.rect.y += 7 * PLAYER_SCALE
                    self.ySpeed = -7
                    self.xSpeed = 0
                else:
                    self.rect.x += 27 * PLAYER_SCALE
                    self.rect.y += 29 * PLAYER_SCALE
                    self.ySpeed = 0
                    self.xSpeed = 7
            else:
                if person.isUp:
                    self.rect.x += 7 * PLAYER_SCALE
                    self.rect.y += 3 * PLAYER_SCALE
                    self.ySpeed = -7
                    self.xSpeed = 0
                else:
                    self.rect.x += -1 * PLAYER_SCALE
                    self.rect.y += 29 * PLAYER_SCALE
                    self.ySpeed = 0
                    self.xSpeed = -7

        elif person.isWalking:
            if person.direction == Direction.RIGHT:
                if person.isUp:
                    self.rect.x += 23 * PLAYER_SCALE
                    self.rect.y += 17 * PLAYER_SCALE
                    self.ySpeed = -7
                    self.xSpeed = 7
                else:
                    self.rect.x += 27 * PLAYER_SCALE
                    self.rect.y += 29 * PLAYER_SCALE
                    self.ySpeed = 0
                    self.xSpeed = 7
            else:
                if person.isUp:
                    self.rect.x += -3 * PLAYER_SCALE
                    self.rect.y += -1 * PLAYER_SCALE
                    self.ySpeed = -7
                    self.xSpeed = -7
                else:
                    self.rect.x += -1 * PLAYER_SCALE
                    self.rect.y += 29 * PLAYER_SCALE
                    self.ySpeed = 0
                    self.xSpeed = -7

    def move(self):
        self.xSpeed += self.xAcc
        self.ySpeed += self.yAcc
        self.rect.x += self.xSpeed
        self.rect.y += self.ySpeed
        self.checkBullet()

    def draw(self, window):
        window.blit(self.image, self.rect)

    def checkBullet(self):
        toDestroy = False
        if self.rect.top < 0 or self.rect.top > 600:
            toDestroy = True
        if self.rect.left < 0 or self.rect.right > 900:
            toDestroy = True
        if toDestroy:
            self.isDestroy = True

    def collideEnemy(self, enemyList, explodeList):
        for enemy in enemyList:
            if pygame.sprite.collide_rect(self, enemy):
                if enemy.type == 3 or enemy.type == 4:
                    enemy.life -= 1
                    Sound('../Sound/hitWeakness.mp3').play()
                    if enemy.life <= 0:
                        self.isDestroy = True
                        enemy.isDestroy = True
                        explodeList.append(Explode(enemy, ExplodeVariety.BRIDGE))
                else:
                    Sound('../Sound/enemyDie.mp3').play()
                    self.isDestroy = True
                    enemy.isDestroy = True
                    explodeList.append(Explode(enemy))

    def collidePlayer(self, player, explodeList):
        if pygame.sprite.collide_rect(self, player):
            # 蹲下的时候，由于图片上半部分是空白，所有子弹必须击中下半部分，才判断为玩家被击中
            if player.isDown or player.isSquating:
                x = player.rect.x
                y = player.rect.y + player.rect.height / 2 + 5
                if (x < self.rect.x < player.rect.x + player.rect.width) and (y < self.rect.y < player.rect.y + player.rect.height):
                    if player.damage(1):
                        self.isDestroy = True
                        explodeList.append(Explode(player, ExplodeVariety.PLAYER1))
                        return True
            elif player.isInWater:
                x = player.rect.x
                y = player.rect.y + player.rect.height / 2
                if (x < self.rect.x < player.rect.x + player.rect.width) and (
                        y < self.rect.y < player.rect.y + player.rect.height):
                    if player.damage(1):
                        self.isDestroy = True
                        explodeList.append(Explode(player, ExplodeVariety.PLAYER1))
                        return True
            else:
                if player.damage(1):
                    self.isDestroy = True
                    explodeList.append(Explode(player, ExplodeVariety.PLAYER1))
                    return True
        return False

    def collideLand(self, enemyLandGroup, explodeList):
        # 如果子弹是由enemyType为3的敌人发射的，就要检测地面碰撞，否则不检测
        if self.enemyType == 3:
            for land in enemyLandGroup:
                if pygame.sprite.collide_rect(self, land):
                    self.isDestroy = True
                    # 让子弹爆炸的位置上移20像素，这样爆炸效果不会显示到陆地下面
                    self.rect.y -= 20
                    explodeList.append(Explode(self, ExplodeVariety.BRIDGE))
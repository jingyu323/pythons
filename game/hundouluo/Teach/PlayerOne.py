
from Constants import *
from Bullet import Bullet
from Sound import Sound

class PlayerOne(pygame.sprite.Sprite):

    def __init__(self, currentTime, life):
        pygame.sprite.Sprite.__init__(self)
        # 加载角色图片
        self.standRightImage = loadImage('../Image/Player/Player1/Right/stand.png')
        self.standLeftImage = loadImage('../Image/Player/Player1/Left/stand.png')
        self.upRightImage = loadImage('../Image/Player/Player1/Up/upRight(small).png')
        self.upLeftImage = loadImage('../Image/Player/Player1/Up/upLeft(small).png')
        self.downRightImage = loadImage('../Image/Player/Player1/Down/down.png')
        self.downLeftImage = loadImage('../Image/Player/Player1/Down/down.png', True)
        self.obliqueUpRightImages = [
            loadImage('../Image/Player/Player1/Up/rightUp1.png'),
            loadImage('../Image/Player/Player1/Up/rightUp2.png'),
            loadImage('../Image/Player/Player1/Up/rightUp3.png'),
        ]
        self.obliqueUpLeftImages = [
            loadImage('../Image/Player/Player1/Up/rightUp1.png', True),
            loadImage('../Image/Player/Player1/Up/rightUp2.png', True),
            loadImage('../Image/Player/Player1/Up/rightUp3.png', True),
        ]
        self.obliqueDownRightImages = [
            loadImage('../Image/Player/Player1/ObliqueDown/1.png'),
            loadImage('../Image/Player/Player1/ObliqueDown/2.png'),
            loadImage('../Image/Player/Player1/ObliqueDown/3.png'),
        ]
        self.obliqueDownLeftImages = [
            loadImage('../Image/Player/Player1/ObliqueDown/1.png', True),
            loadImage('../Image/Player/Player1/ObliqueDown/2.png', True),
            loadImage('../Image/Player/Player1/ObliqueDown/3.png', True),
        ]
        # 角色向右的全部图片
        self.rightImages = [
            loadImage('../Image/Player/Player1/Right/run1.png'),
            loadImage('../Image/Player/Player1/Right/run2.png'),
            loadImage('../Image/Player/Player1/Right/run3.png')
        ]
        # 角色向左的全部图片
        self.leftImages = [
            loadImage('../Image/Player/Player1/Left/run1.png'),
            loadImage('../Image/Player/Player1/Left/run2.png'),
            loadImage('../Image/Player/Player1/Left/run3.png')
        ]
        # 角色跳跃的全部图片
        self.upRightImages = [
            loadImage('../Image/Player/Player1/Jump/jump1.png'),
            loadImage('../Image/Player/Player1/Jump/jump2.png'),
            loadImage('../Image/Player/Player1/Jump/jump3.png'),
            loadImage('../Image/Player/Player1/Jump/jump4.png'),
        ]
        self.upLeftImages = [
            loadImage('../Image/Player/Player1/Jump/jump1.png', True),
            loadImage('../Image/Player/Player1/Jump/jump2.png', True),
            loadImage('../Image/Player/Player1/Jump/jump3.png', True),
            loadImage('../Image/Player/Player1/Jump/jump4.png', True),
        ]
        self.rightFireImages = [
            loadImage('../Image/Player/Player1/Right/fire1.png'),
            loadImage('../Image/Player/Player1/Right/fire2.png'),
            loadImage('../Image/Player/Player1/Right/fire3.png'),
        ]
        self.leftFireImages = [
            loadImage('../Image/Player/Player1/Right/fire1.png', True),
            loadImage('../Image/Player/Player1/Right/fire2.png', True),
            loadImage('../Image/Player/Player1/Right/fire3.png', True),
        ]
        # 加载玩家在水中的图片
        self.upRightImageInWater = loadImage('../Image/Player/Player1/Water/up.png')
        self.upLeftImageInWater = loadImage('../Image/Player/Player1/Water/up.png', True)
        self.diveRightImageInWater = loadImage('../Image/Player/Player1/Water/dive.png')
        self.diveLeftImageInWater = loadImage('../Image/Player/Player1/Water/dive.png', True)
        self.standRightImageInWater = loadImage('../Image/Player/Player1/Water/stand.png')
        self.standLeftImageInWater = loadImage('../Image/Player/Player1/Water/stand.png', True)
        self.fireRightInWater = loadImage('../Image/Player/Player1/Water/standFire.png')
        self.fireLeftInWater = loadImage('../Image/Player/Player1/Water/standFire.png', True)
        self.obliqueRightInWater = loadImage('../Image/Player/Player1/Water/obliqueRight.png')
        self.obliqueLeftInWater = loadImage('../Image/Player/Player1/Water/obliqueRight.png', True)
        self.rightInWaterImage = loadImage('../Image/Player/Player1/Water/inWater.png')
        self.leftInWaterImage = loadImage('../Image/Player/Player1/Water/inWater.png', True)
        # 角色左右移动下标
        self.imageIndex = 0
        # 角色跳跃下标
        self.upImageIndex = 0
        # 角色斜射下标
        self.obliqueImageIndex = 0
        # 上一次显示图片的时间
        self.runLastTimer = currentTime
        self.fireLastTimer = currentTime

        # 选择当前要显示的图片
        self.image = self.standRightImage
        # 获取图片的rect
        self.rect = self.image.get_rect()
        # 设置角色的状态
        self.state = State.FALL
        # 角色的方向
        self.direction = Direction.RIGHT
        # 速度
        self.xSpeed = PLAYER_X_SPEED
        self.ySpeed = 0
        self.jumpSpeed = -11
        # 人物当前的状态标志
        self.isStanding = False
        self.isWalking = False
        self.isJumping = True
        self.isSquating = False
        self.isFiring = False
        self.isInWater = False
        # 重力加速度
        self.gravity = 0.8

        # 玩家上下方向
        self.isUp = False
        self.isDown = False

        self.life = life
        self.isInvincible = True

    def update(self, keys, currentTime, playerBulletList):
        # 更新站或者走的状态
        # 根据状态响应按键
        if self.state == State.STAND:
            self.standing(keys, currentTime, playerBulletList)
        elif self.state == State.WALK:
            self.walking(keys, currentTime, playerBulletList)
        elif self.state == State.JUMP:
            self.jumping(keys, currentTime, playerBulletList)
        elif self.state == State.FALL:
            self.falling(keys, currentTime, playerBulletList)

        # 更新动画
        if self.isInWater:
            self.waterUpdate()
        else:
            self.landUpdate()

    def landUpdate(self):
        # 跳跃状态
        if self.isJumping:
            # 根据方向
            if self.direction == Direction.RIGHT:
                # 方向向右，角色加载向右跳起的图片
                self.image = self.upRightImages[self.upImageIndex]
            else:
                # 否则，方向向左，角色加载向左跳起的图片
                self.image = self.upLeftImages[self.upImageIndex]

        # 角色蹲下
        if self.isSquating:
            if self.direction == Direction.RIGHT:
                # 加载向右蹲下的图片
                self.image = self.downRightImage
            else:
                # 加载向左蹲下的图片
                self.image = self.downLeftImage

        # 角色站着
        if self.isStanding:
            if self.direction == Direction.RIGHT:
                if self.isUp:
                    # 加载向右朝上的图片
                    self.image = self.upRightImage
                elif self.isDown:
                    # 加载向右蹲下的图片
                    self.image = self.downRightImage
                else:
                    # 加载向右站着的图片
                    self.image = self.standRightImage
            else:
                # 向左也是同样的效果
                if self.isUp:
                    self.image = self.upLeftImage
                elif self.isDown:
                    self.image = self.downLeftImage
                else:
                    self.image = self.standLeftImage

        # 角色移动
        if self.isWalking:
            if self.direction == Direction.RIGHT:
                if self.isUp:
                    # 加载斜右上的图片
                    self.image = self.obliqueUpRightImages[self.obliqueImageIndex]
                elif self.isDown:
                    # 加载斜右下的图片
                    self.image = self.obliqueDownRightImages[self.obliqueImageIndex]
                else:
                    # 加载向右移动的图片，根据开火状态是否加载向右开火移动的图片
                    if self.isFiring:
                        self.image = self.rightFireImages[self.imageIndex]
                    else:
                        self.image = self.rightImages[self.imageIndex]
            else:
                if self.isUp:
                    self.image = self.obliqueUpLeftImages[self.obliqueImageIndex]
                elif self.isDown:
                    self.image = self.obliqueDownLeftImages[self.obliqueImageIndex]
                else:
                    if self.isFiring:
                        self.image = self.leftFireImages[self.imageIndex]
                    else:
                        self.image = self.leftImages[self.imageIndex]

    def waterUpdate(self):
        if self.isSquating:
            if self.direction == Direction.RIGHT:
                self.image = self.diveRightImageInWater
            else:
                self.image = self.diveLeftImageInWater

        if self.isStanding:
            if self.direction == Direction.RIGHT:
                if self.isFiring:
                    if self.isUp:
                        self.image = self.upRightImageInWater
                    else:
                        self.image = self.fireRightInWater
                else:
                    if self.isUp:
                        self.image = self.upRightImageInWater
                    else:
                        self.image = self.standRightImageInWater
            else:
                if self.isFiring:
                    if self.isUp:
                        self.image = self.upLeftImageInWater
                    else:
                        self.image = self.fireLeftInWater
                else:
                    if self.isUp:
                        self.image = self.upLeftImageInWater
                    else:
                        self.image = self.standLeftImageInWater

        if self.isWalking:
            if self.direction == Direction.RIGHT:
                if self.isUp:
                    self.image = self.obliqueRightInWater
                else:
                    if self.isFiring:
                        self.image = self.fireRightInWater
                    else:
                        self.image = self.standRightImageInWater
            else:
                if self.isUp:
                    self.image = self.obliqueLeftInWater
                else:
                    if self.isFiring:
                        self.image = self.fireLeftInWater
                    else:
                        self.image = self.standLeftImageInWater

    def standing(self, keys, currentTime, playerBulletList):
        """角色站立"""

        # 设置角色状态
        self.isStanding = True
        self.isWalking = False
        self.isJumping = False
        self.isSquating = False
        self.isUp = False
        self.isDown = False
        self.isFiring = False

        # 设置速度
        self.ySpeed = 0
        self.xSpeed = 0

        # 按下A键
        if keys[pygame.K_a]:
            # A按下，角色方向向左
            self.direction = Direction.LEFT
            # 改变角色的状态，角色进入移动状态
            self.state = State.WALK
            # 设置站立状态为False，移动状态为True
            self.isStanding = False
            self.isWalking = True
            # 向左移动，速度为负数，这样玩家的x坐标是减小的
            self.xSpeed = -PLAYER_X_SPEED
        # 按下D键
        elif keys[pygame.K_d]:
            # D按下，角色方向向右
            self.direction = Direction.RIGHT
            # 改变角色的状态，角色进入移动状态
            self.state = State.WALK
            # 设置站立状态为False，移动状态为True
            self.isStanding = False
            self.isWalking = True
            # 向右移动，速度为正数
            self.xSpeed = PLAYER_X_SPEED
        # 按下k键
        elif keys[pygame.K_k]:
            if not self.isInWater:
                # K按下，角色进入跳跃状态，但是不会改变方向
                self.state = State.JUMP
                # 设置站立状态为False，跳跃状态为True
                # 不改变移动状态，因为移动的时候也可以跳跃
                self.isStanding = False
                self.isJumping = True
                # 设置速度，速度为负数，因为角色跳起后，要下落
                self.isUp = True
                self.ySpeed = self.jumpSpeed
        # 没有按下按键
        else:
            # 没有按下按键，角色依然是站立状态
            self.state = State.STAND
            self.isStanding = True

        # 按下w键
        if keys[pygame.K_w]:
            # W按下，角色向上，改变方向状态
            self.isUp = True
            self.isStanding = True
            self.isDown = False
            self.isSquating = False
        # 按下s键
        elif keys[pygame.K_s]:
            # S按下，角色蹲下，改变方向状态，并且蹲下状态设置为True
            self.isUp = False
            self.isStanding = False
            self.isDown = True
            self.isSquating = True

        if keys[pygame.K_j]:
            self.fire(currentTime, playerBulletList)

    def walking(self, keys, currentTime, playerBulletList):
        """角色行走，每10帧变换一次图片"""
        self.isStanding = False
        self.isWalking = True
        self.isJumping = False
        self.isSquating = False
        self.isFiring = False
        self.ySpeed = 0
        self.xSpeed = PLAYER_X_SPEED

        if self.isInWater:
            self.walkingInWater(currentTime)
        else:
            self.walkingInLand(currentTime)

        # 按下D键
        if keys[pygame.K_d]:
            self.direction = Direction.RIGHT
            self.xSpeed = PLAYER_X_SPEED
        # 按下A键
        elif keys[pygame.K_a]:
            self.direction = Direction.LEFT
            self.xSpeed = -PLAYER_X_SPEED
         # 按下S键
        elif keys[pygame.K_s]:
            self.isStanding = False
            self.isDown = True
            self.isUp = False

        # 按下W键
        if keys[pygame.K_w]:
            self.isUp = True
            self.isDown = False
        # 没有按键按下
        else:
            self.state = State.STAND

        # 移动时按下K键
        if keys[pygame.K_k]:
            # 角色状态变为跳跃
            if not self.isInWater:
                self.state = State.JUMP
                self.ySpeed = self.jumpSpeed
                self.isJumping = True
                self.isStanding = False
                self.isUp = True

        if keys[pygame.K_j]:
            self.fire(currentTime, playerBulletList)

    def walkingInLand(self, currentTime):
        # 如果当前是站立的图片
        if self.isStanding:
            # 方向向右，方向向上
            if self.direction == Direction.RIGHT and self.isUp:
                # 设置为向右朝上的图片
                self.image = self.upRightImage
            # 方向向右
            elif self.direction == Direction.RIGHT and not self.isUp:
                # 设置为向右站立的图片
                self.image = self.standRightImage
            elif self.direction == Direction.LEFT and self.isUp:
                self.image = self.upLeftImage
            elif self.direction == Direction.LEFT and not self.isUp:
                self.image = self.standLeftImage
            # 记下当前时间
            self.runLastTimer = currentTime
        else:
            # 如果是走动的图片，先判断方向
            if self.direction == Direction.RIGHT:
                # 设置速度
                self.xSpeed = PLAYER_X_SPEED
                # 根据上下方向觉得是否角色要加载斜射的图片
                if self.isUp or self.isDown:
                    # isUp == True表示向上斜射
                    # isDown == True表示向下斜射
                    # 计算上一次加载图片到这次的时间，如果大于115，即11.5帧，即上次加载图片到这次加载图片之间，已经加载了11张图片
                    if currentTime - self.runLastTimer > 115:
                        # 那么就可以加载斜着奔跑的图片
                        # 如果角色加载的图片不是第三张，则加载下一张就行
                        if self.obliqueImageIndex < 2:
                            self.obliqueImageIndex += 1
                        # 否则就加载第一张图片
                        else:
                            self.obliqueImageIndex = 0
                        # 记录变换图片的时间，为下次变换图片做准备
                        self.runLastTimer = currentTime
                # 不是斜射
                else:
                    # 加载正常向右奔跑的图片
                    if currentTime - self.runLastTimer > 115:
                        if self.imageIndex < 2:
                            self.imageIndex += 1
                        else:
                            self.imageIndex = 0
                        self.runLastTimer = currentTime
            else:
                self.xSpeed = -PLAYER_X_SPEED
                if self.isUp or self.isDown:
                    if currentTime - self.runLastTimer > 115:
                        if self.obliqueImageIndex < 2:
                            self.obliqueImageIndex += 1
                        else:
                            self.obliqueImageIndex = 0
                        self.runLastTimer = currentTime
                else:
                    if currentTime - self.runLastTimer > 115:
                        if self.imageIndex < 2:
                            self.imageIndex += 1
                        else:
                            self.imageIndex = 0
                        self.runLastTimer = currentTime

    def walkingInWater(self, currentTime):
        if self.isStanding:
            # 设置为斜射
            if self.direction == Direction.RIGHT and self.isUp:
                self.image = self.upRightImageInWater
            elif self.direction == Direction.RIGHT and not self.isUp:
                self.image = self.standRightImageInWater
            elif self.direction == Direction.LEFT and self.isUp:
                self.image = self.upLeftImageInWater
            elif self.direction == Direction.LEFT and not self.isUp:
                self.image = self.standLeftImageInWater
            self.runLastTimer = currentTime
        else:
            # 如果是走动的图片
            if self.direction == Direction.RIGHT:
                self.xSpeed = PLAYER_X_SPEED
                if self.isUp:
                    self.image = self.obliqueRightInWater
                    self.runLastTimer = currentTime
                else:
                    self.image = self.standRightImageInWater
                    self.runLastTimer = currentTime
            else:
                self.xSpeed = PLAYER_X_SPEED
                if self.isUp:
                    self.image = self.obliqueLeftInWater
                    self.runLastTimer = currentTime
                else:
                    self.image = self.standLeftImageInWater
                    self.runLastTimer = currentTime

    def jumping(self, keys, currentTime, playerBulletList):
        """跳跃"""
        # 设置标志
        self.isJumping = True
        self.isStanding = False
        self.isDown = False
        self.isSquating = False
        self.isFiring = False
        # 更新速度
        self.ySpeed += self.gravity
        if currentTime - self.runLastTimer > 115:
            if self.upImageIndex < 3:
                self.upImageIndex += 1
            else:
                self.upImageIndex = 0
            # 记录变换图片的时间，为下次变换图片做准备
            self.runLastTimer = currentTime

        if keys[pygame.K_d]:
            self.direction = Direction.RIGHT

        elif keys[pygame.K_a]:
            self.direction = Direction.LEFT

        # 按下W键
        if keys[pygame.K_w]:
            self.isUp = True
            self.isDown = False
        elif keys[pygame.K_s]:
            self.isUp = False
            self.isDown = True

        if self.ySpeed >= 0:
            self.state = State.FALL

        if not keys[pygame.K_k]:
            self.state = State.FALL

        if keys[pygame.K_j]:
            self.fire(currentTime, playerBulletList)

    def falling(self, keys, currentTime, playerBulletList):
        # 下落时速度越来越快，所以速度需要一直增加
        self.ySpeed += self.gravity
        if currentTime - self.runLastTimer > 115:
            if self.upImageIndex < 3:
                self.upImageIndex += 1
            else:
                self.upImageIndex = 0
            self.runLastTimer = currentTime

        if keys[pygame.K_d]:
            self.direction = Direction.RIGHT
            self.isWalking = False

        elif keys[pygame.K_a]:
            self.direction = Direction.LEFT
            self.isWalking = False

        if keys[pygame.K_j]:
            self.fire(currentTime, playerBulletList)

    def fire(self, currentTime, playerBulletList):
        self.isFiring = True
        # 潜水状态下不能开火
        if not (self.isInWater and self.isSquating):
            if len(playerBulletList) < PLAYER_BULLET_NUMBER:
                if currentTime - self.fireLastTimer > 150:
                    Sound('../Sound/commonFire.mp3').play()
                    playerBulletList.append(Bullet(self))
                    self.fireLastTimer = currentTime

    def damage(self, damage):
        if not self.isInvincible:
            self.life -= damage
            return True
        return False

    def getCenter(self):
        return self.rect.x + self.rect.width / 2, self.rect.y + self.rect.height / 2 + y0
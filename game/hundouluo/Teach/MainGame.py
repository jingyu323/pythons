import copy
import sys
import pygame
from Constants import *
from PlayerOne import PlayerOne
from Collider import Collider
from Enemy1 import Enemy1
from Explode import Explode
from Enemy2 import Enemy2
from Bridge import Bridge
from BossPart import BossPart
from Sound import Sound
import time

def drawPlayerOneBullet(player1BulletList):
    for bullet in player1BulletList:
        if bullet.isDestroy:
            player1BulletList.remove(bullet)
        else:
            bullet.draw(MainGame.window)
            bullet.move()
            bullet.collideEnemy(MainGame.enemyList, MainGame.explodeList)


def enemyUpdate(enemyList, enemyBulletList):
    # 遍历整个敌人列表
    for enemy in enemyList:
        if enemy.type == 1:
            if enemy.isDestroy:
                enemyList.remove(enemy)
                MainGame.allSprites.remove(enemy)
                MainGame.enemyGroup.remove(enemy)
            else:
                enemy.checkPosition(MainGame.player1.rect.x, MainGame.player1.rect.y)
                enemy.draw(pygame.time.get_ticks())
                enemy.move(pygame.time.get_ticks())
                enemy.fire(enemyBulletList)
        elif enemy.type == 2:
            if enemy.isDestroy:
                enemyList.remove(enemy)
                MainGame.allSprites.remove(enemy)
                MainGame.enemyGroup.remove(enemy)
            else:
                enemy.checkPosition(MainGame.player1.rect.x, MainGame.player1.rect.y)
                enemy.draw(MainGame.window, MainGame.player1, pygame.time.get_ticks())
                enemy.fire(enemyBulletList, MainGame.player1)


def updateEnemyPosition():
    # 遍历全部敌人列表
    for enemy in MainGame.enemyList:
        if enemy.type == 1:
            # 创建一个复制
            t = copy.copy(enemy)
            t.rect.y += 1
            # 让复制的y加1，看看有没有发生碰撞，这里看的碰撞是enemyColliderGroup和commonColliderGroup中的碰撞
            collide = pygame.sprite.spritecollideany(t, MainGame.enemyColliderGroup) \
                      or pygame.sprite.spritecollideany(t, MainGame.commonColliderGroup)
            # 没有发生碰撞，让敌人下落
            if not collide:
                enemy.rect.y += 4
                enemy.isFalling = True
                # 改变下落时的图片
                enemy.image = enemy.rightFallImage if enemy.direction == Direction.RIGHT else enemy.leftFallImage
            else:
                enemy.isFalling = False
                # 如果与河发生碰撞，表示敌人落到了水中，那么敌人直接死亡
                if collide in MainGame.enemyRiverGroup:
                    enemy.isDestroy = True
                    MainGame.explodeList.append(Explode(enemy))
            t.rect.y -= 1
        elif enemy.type == 2:
            t = copy.copy(enemy)
            t.rect.y += 1
            collide = pygame.sprite.spritecollideany(t, MainGame.enemyColliderGroup)
            if not collide:
                enemy.rect.y += 1
            t.rect.y -= 1
        elif enemy.type == 3:
            if enemy.isDestroy:
                enemyList.remove(enemy)
                MainGame.allSprites.remove(enemy)
                MainGame.enemyGroup.remove(enemy)
            else:
                enemy.fire(pygame.time.get_ticks(), MainGame.enemyBulletList)
        elif enemy.type == 4:
            # 如果4被消灭，表示玩家第一关通过
            if enemy.isDestroy:
                enemyList.remove(enemy)
                MainGame.allSprites.remove(enemy)
                MainGame.enemyGroup.remove(enemy)
                victory()

def victory():
    pass

def drawEnemyBullet(enemyBulletList):
    for bullet in enemyBulletList:
        if bullet.isDestroy:
            enemyBulletList.remove(bullet)
        else:
            bullet.draw(MainGame.window)
            bullet.move()
            bullet.collideLand(MainGame.enemyLandGroup, MainGame.explodeList)
            if bullet.collidePlayer(MainGame.player1, MainGame.explodeList):
                initPlayer1(MainGame.player1.life)


def initLand():
    land1 = Collider(81, 119 * MAP_SCALE, 737 * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    # land1 = Collider(81, 119 * MAP_SCALE, 8000 * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land2 = Collider(400, 151 * MAP_SCALE, 96 * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land3 = Collider(640, 183 * MAP_SCALE, 33 * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land4 = Collider(880, 183 * MAP_SCALE, 33 * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land5 = Collider(720, 215 * MAP_SCALE, 2 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land6 = Collider(1040, 154 * MAP_SCALE, 2 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land7 = Collider(1600, 166 * MAP_SCALE, 3 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land8 = Collider(1120 * RATIO, 215 * MAP_SCALE, 2 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land9 = Collider(1650 * RATIO, 119 * MAP_SCALE, 5 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land10 = Collider(2185 * RATIO, 119 * MAP_SCALE, 8 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land11 = Collider(2595 * RATIO, 215 * MAP_SCALE, 3 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land12 = Collider(2770 * RATIO, 167 * MAP_SCALE, 2 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land13 = Collider(2535 * RATIO, 87 * MAP_SCALE, 16 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land14 = Collider(2950 * RATIO, 151 * MAP_SCALE, 7 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land15 = Collider(3185 * RATIO, 215 * MAP_SCALE, 6 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land16 = Collider(3420 * RATIO, 119 * MAP_SCALE, 7 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land17 = Collider(3537 * RATIO, 183 * MAP_SCALE, 2 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land18 = Collider(3715 * RATIO, 183 * MAP_SCALE, 2 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land19 = Collider(3890 * RATIO, 167 * MAP_SCALE, 1 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land20 = Collider(3775 * RATIO, 87 * MAP_SCALE, 5 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land21 = Collider(4010 * RATIO, 151 * MAP_SCALE, 3 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land22 = Collider(4125 * RATIO, 119 * MAP_SCALE, 2 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land23 = Collider(4304 * RATIO, 151 * MAP_SCALE, 2 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land24 = Collider(4304 * RATIO, 216 * MAP_SCALE, 1 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land25 = Collider(4361 * RATIO, 183 * MAP_SCALE, 3 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land26 = Collider(4537 * RATIO, 119 * MAP_SCALE, 2 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land27 = Collider(4598 * RATIO, 87 * MAP_SCALE, 2 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land28 = Collider(4657 * RATIO, 167 * MAP_SCALE, 1 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land29 = Collider(4598 * RATIO, 216 * MAP_SCALE, 1 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land30 = Collider(4776 * RATIO, 119 * MAP_SCALE, 2 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land31 = Collider(4835 * RATIO, 151 * MAP_SCALE, 5 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land32 = Collider(5010 * RATIO, 216 * MAP_SCALE, 3 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land33 = Collider(5250 * RATIO, 183 * MAP_SCALE, 2 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land34 = Collider(5423 * RATIO, 151 * MAP_SCALE, 2 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land35 = Collider(5543 * RATIO, 119 * MAP_SCALE, 4 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land36 = Collider(5601 * RATIO, 167 * MAP_SCALE, 3 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land37 = Collider(5541 * RATIO, 216 * MAP_SCALE, 8 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land38 = Collider(5776 * RATIO, 151 * MAP_SCALE, 1 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    land39 = Collider(5836 * RATIO, 183 * MAP_SCALE, 1 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    MainGame.playerLandGroup = pygame.sprite.Group(
        land1, land2, land3, land4, land5, land6, land7, land8, land9, land10,
        land11, land12, land13, land14, land15, land16, land17, land18, land19, land20,
        land21, land22, land23, land24, land25, land26, land27, land28, land29, land30,
        land31, land32, land33, land34, land35, land36, land37, land38, land39
    )
    eland1 = Collider(81, 119 * MAP_SCALE, 737 * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    eland8 = Collider(1120 * RATIO, 215 * MAP_SCALE, 2 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    eland9 = Collider(1650 * RATIO, 119 * MAP_SCALE, 5 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    eland10 = Collider(2185 * RATIO, 119 * MAP_SCALE, 8 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    eland37 = Collider(5541 * RATIO, 216 * MAP_SCALE, 8 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE)
    MainGame.enemyLandGroup = pygame.sprite.Group(eland1, eland8, eland9, eland10, eland37)
    MainGame.playerColliderGroup.add(MainGame.playerLandGroup)
    MainGame.enemyColliderGroup.add(MainGame.enemyLandGroup)


def initRiver():
    river1 = Collider(0, 215 * MAP_SCALE, 289 * MAP_SCALE, LAND_THICKNESS * MAP_SCALE, (0, 0, 255))
    river2 = Collider(880, 215 * MAP_SCALE, 255 * MAP_SCALE, LAND_THICKNESS * MAP_SCALE, (0, 0, 255))
    river3 = Collider(1680, 215 * MAP_SCALE, 737 * MAP_SCALE, LAND_THICKNESS * MAP_SCALE, (0, 0, 255))
    eRiver1 = Collider(0, 215 * MAP_SCALE, 289 * MAP_SCALE, LAND_THICKNESS * MAP_SCALE, (0, 0, 255))
    eRiver3 = Collider(1680, 215 * MAP_SCALE, 737 * MAP_SCALE, LAND_THICKNESS * MAP_SCALE, (0, 0, 255))
    MainGame.playerRiverGroup = pygame.sprite.Group(river1, river2, river3)
    MainGame.enemyRiverGroup = pygame.sprite.Group(eRiver1, eRiver3)
    MainGame.playerColliderGroup.add(MainGame.playerRiverGroup)
    MainGame.enemyColliderGroup.add(MainGame.enemyRiverGroup)


def drawExplode(explodeList):
    for explode in explodeList:
        if explode.isDestroy:
            explodeList.remove(explode)
        else:
            if explode.isUseTime:
                explode.draw(MainGame.window, pygame.time.get_ticks())
            else:
                explode.draw(MainGame.window)


def drawBridge(bridgeList):
    for b in bridgeList:
        if b.isDestroy:
            bridgeList.remove(b)
        else:
            b.draw(MainGame.window)


def initPlayer1(life):
    if life == 0:
        pass
    MainGame.allSprites.remove(MainGame.player1)
    MainGame.player1 = PlayerOne(pygame.time.get_ticks(), life)
    MainGame.player1.rect.x = 80
    MainGame.player1.rect.bottom = 0
    # 把角色放入组中，方便统一管理
    MainGame.allSprites.add(MainGame.player1)


def generateEnemy1(x, y, direction, currentTime):
    # 根据玩家的当前位置和方向产生一个敌人
    enemy = Enemy1(x, y, direction, currentTime)
    # 分别加入敌人列表，所有角色组，敌人碰撞组
    MainGame.enemyList.append(enemy)
    MainGame.allSprites.add(enemy)
    MainGame.enemyGroup.add(enemy)


def generateEnemy2(x, y):
    enemy = Enemy2(x, y, MainGame.player1, pygame.time.get_ticks())
    MainGame.enemyList.append(enemy)
    MainGame.allSprites.add(enemy)
    MainGame.enemyGroup.add(enemy)


def initBridge():
    bridge1_1 = Bridge(1920, int(113 * MAP_SCALE), BridgeType.ON)
    bridge1_2 = Bridge(1980, int(113 * MAP_SCALE), BridgeType.BODY)
    bridge1_3 = Bridge(2040, int(113 * MAP_SCALE), BridgeType.BODY)
    bridge1_4 = Bridge(2100, int(113 * MAP_SCALE), BridgeType.BODY)
    bridge1_5 = Bridge(2160, int(113 * MAP_SCALE), BridgeType.BODY)
    bridge1_6 = Bridge(2180, int(113 * MAP_SCALE), BridgeType.DOWN)
    bridge2_1 = Bridge(2640, int(113 * MAP_SCALE), BridgeType.ON)
    bridge2_2 = Bridge(2700, int(113 * MAP_SCALE), BridgeType.BODY)
    bridge2_3 = Bridge(2760, int(113 * MAP_SCALE), BridgeType.BODY)
    bridge2_4 = Bridge(2820, int(113 * MAP_SCALE), BridgeType.BODY)
    bridge2_5 = Bridge(2880, int(113 * MAP_SCALE), BridgeType.BODY)
    bridge2_6 = Bridge(2900, int(113 * MAP_SCALE), BridgeType.DOWN)
    bridgeCollide1_1 = Collider(1920, 119 * MAP_SCALE, 1 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE,
                                (0, 255, 0))
    bridgeCollide1_2 = Collider(1980, 119 * MAP_SCALE, 1 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE,
                                (0, 255, 0))
    bridgeCollide1_3 = Collider(2040, 119 * MAP_SCALE, 1 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE,
                                (0, 255, 0))
    bridgeCollide1_4 = Collider(2100, 119 * MAP_SCALE, 1 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE,
                                (0, 255, 0))
    bridgeCollide1_5 = Collider(2160, 119 * MAP_SCALE, 1 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE,
                                (0, 255, 0))
    bridgeCollide1_6 = Collider(2180, 119 * MAP_SCALE, 0.8 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE,
                                (0, 255, 0))
    bridgeCollide2_1 = Collider(2640, 119 * MAP_SCALE, 1 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE,
                                (0, 255, 0))
    bridgeCollide2_2 = Collider(2700, 119 * MAP_SCALE, 1 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE,
                                (0, 255, 0))
    bridgeCollide2_3 = Collider(2760, 119 * MAP_SCALE, 1 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE,
                                (0, 255, 0))
    bridgeCollide2_4 = Collider(2820, 119 * MAP_SCALE, 1 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE,
                                (0, 255, 0))
    bridgeCollide2_5 = Collider(2880, 119 * MAP_SCALE, 1 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE,
                                (0, 255, 0))
    bridgeCollide2_6 = Collider(2900, 119 * MAP_SCALE, 0.8 * LAND_LENGTH * MAP_SCALE, LAND_THICKNESS * MAP_SCALE,
                                (0, 255, 0))
    MainGame.bridgeList.append(bridge1_1)
    MainGame.bridgeList.append(bridge1_2)
    MainGame.bridgeList.append(bridge1_3)
    MainGame.bridgeList.append(bridge1_4)
    MainGame.bridgeList.append(bridge1_5)
    MainGame.bridgeList.append(bridge1_6)
    MainGame.bridgeList.append(bridge2_1)
    MainGame.bridgeList.append(bridge2_2)
    MainGame.bridgeList.append(bridge2_3)
    MainGame.bridgeList.append(bridge2_4)
    MainGame.bridgeList.append(bridge2_5)
    MainGame.bridgeList.append(bridge2_6)
    MainGame.commonColliderGroup = pygame.sprite.Group(
        bridgeCollide1_1, bridgeCollide1_2, bridgeCollide1_3, bridgeCollide1_4, bridgeCollide1_5, bridgeCollide1_6,
        bridgeCollide2_1, bridgeCollide2_2, bridgeCollide2_3, bridgeCollide2_4, bridgeCollide2_5, bridgeCollide2_6,
    )

def endGame():
    MainGame.isEnd = True

class MainGame:
    player1 = None
    allSprites = pygame.sprite.Group()
    
    # 敌人
    enemyList = []
    
    window = None
    # 子弹
    player1BulletList = []
    enemyBulletList = []
    bridgeList = []
    # 爆炸效果
    explodeList = []
    
    # 冲突
    playerLandGroup = pygame.sprite.Group()
    playerRiverGroup = pygame.sprite.Group()
    enemyLandGroup = pygame.sprite.Group()
    enemyRiverGroup = pygame.sprite.Group()
    playerColliderGroup = pygame.sprite.Group()
    enemyColliderGroup = pygame.sprite.Group()
    enemyGroup = pygame.sprite.Group()
    bridgeGroup = pygame.sprite.Group()
    commonColliderGroup = None
    # 冲突栈
    colliderStack = []
    
    # 是否结束游戏
    isEnd = False
    
    def __init__(self):
        
        # 设置成员变量
        self.boss = []
        self.background = None
        self.backRect = None
        self.bridgeExploding = False
        self.enemyBoolList = [True for _ in range(5)]
        
        # 初始化展示模块
        pygame.display.init()
        
        SCREEN_SIZE = (SCREEN_WIDTH, SCREEN_HEIGHT)
        # 初始化窗口
        MainGame.window = pygame.display.set_mode(SCREEN_SIZE)
        # 设置窗口标题
        pygame.display.set_caption('魂斗罗角色')

        # 获取按键
        self.keys = pygame.key.get_pressed()
        # 帧率
        self.fps = 60
        self.clock = pygame.time.Clock()
        
        # 角色
        initPlayer1(3)
        
        # Boss
        self.initBoss()
        
        # 加载背景
        self.initBackground()
        
        # 摄像头调整
        self.cameraAdaption = 0
        
        # 加载场景景物
        initLand()
        initRiver()
        initBridge()
        
        # 碰撞失效间隔
        self.index = 0
        
        # 显示玩家生命值
        self.lifeImage = loadImage('../Image/Player/Player1/Life/life.png')
        
        self.hasCollidedBridge = []
        
        # 加载音乐
        self.backgroundMusic = Sound('../Sound/background.mp3')
    
    def run(self):
        # 播放背景音乐
        self.backgroundMusic.play()
        
        while not self.isEnd:
            
            # 设置背景颜色
            pygame.display.get_surface().fill((0, 0, 0))
            
            # 游戏场景和景物更新函数
            self.update(MainGame.window, MainGame.player1BulletList)
            
            # 获取窗口中的事件
            self.getPlayingModeEvent()
            
            # 更新窗口
            pygame.display.update()
            
            # 设置帧率
            self.clock.tick(self.fps)
            fps = self.clock.get_fps()
            caption = '魂斗罗 - {:.2f}'.format(fps)
            pygame.display.set_caption(caption)
        else:
            sys.exit()
    
    def getPlayingModeEvent(self):
        # 获取事件列表
        for event in pygame.event.get():
            # 点击窗口关闭按钮
            if event.type == pygame.QUIT:
                self.isEnd = True
            # 键盘按键按下
            elif event.type == pygame.KEYDOWN:
                self.keys = pygame.key.get_pressed()
            # 键盘按键抬起
            elif event.type == pygame.KEYUP:
                self.keys = pygame.key.get_pressed()
    
    def update(self, window, player1BulletList):
        # 加载背景
        window.blit(self.background, self.backRect)
        
        # 显示生命图标
        self.drawLifeImage(MainGame.window)
        
        # 加载桥
        drawBridge(MainGame.bridgeList)
        self.bridgeExplode()
        
        # 敌人更新
        enemyUpdate(MainGame.enemyList, MainGame.enemyBulletList)
        drawExplode(MainGame.explodeList)
        drawPlayerOneBullet(MainGame.player1BulletList)
        drawEnemyBullet(MainGame.enemyBulletList)
        # 更新人物
        currentTime = pygame.time.get_ticks()
        MainGame.allSprites.update(self.keys, currentTime, player1BulletList)
        self.updatePlayerPosition()
        updateEnemyPosition()
        # 摄像机移动
        self.camera()
        # 显示物体
        MainGame.allSprites.draw(window)
        # 加载敌人
        self.generateEnemy()
        
        for collider in MainGame.playerLandGroup:
            r = collider.draw(window, self.player1.rect.y)
            # 如果没有画出来，表示玩家高度低于直线，所有把直线从组中删除
            if not r:
                # 删除前先检查一下是不是在组中
                if collider in MainGame.playerColliderGroup:
                    # 删除并加入栈
                    MainGame.colliderStack.insert(0, collider)
                    MainGame.playerColliderGroup.remove(collider)
            else:
                # 如果画出来了，判断一下玩家距离是否高于线的距离
                if collider.rect.y > self.player1.rect.bottom:
                    # 如果是的话，且冲突栈不为空，那么从栈中取出一个元素放入冲突组，最前面的元素一定是先如队列的
                    if len(MainGame.colliderStack) > 0:
                        f = MainGame.colliderStack.pop()
                        MainGame.playerColliderGroup.add(f)
        MainGame.playerRiverGroup.draw(window)
        MainGame.commonColliderGroup.draw(window)
    
    def camera(self):
        # 如果玩家的右边到达了屏幕的一半
        if self.player1.rect.right > SCREEN_WIDTH / 2:
            if not (self.backRect.x <= -3500 * MAP_SCALE):
                # 计算出超过的距离
                self.cameraAdaption = self.player1.rect.right - SCREEN_WIDTH / 2
                # 让背景向右走这么多距离
                self.backRect.x -= self.cameraAdaption
                # 场景中的物体都走这么多距离
                self.mapObjectMove()
    
    def mapObjectMove(self):
        for sprite in MainGame.allSprites:
            sprite.rect.x -= self.cameraAdaption
        for collider in MainGame.playerColliderGroup:
            collider.rect.x -= self.cameraAdaption
        for collider in MainGame.colliderStack:
            collider.rect.x -= self.cameraAdaption
        for collider in MainGame.enemyColliderGroup:
            collider.rect.x -= self.cameraAdaption
        for collider in MainGame.commonColliderGroup:
            collider.rect.x -= self.cameraAdaption
        for bridge in MainGame.bridgeList:
            bridge.rect.x -= self.cameraAdaption
        for explode in MainGame.explodeList:
            explode.rect.x -= self.cameraAdaption
        for bullet in MainGame.enemyBulletList:
            bullet.rect.x -= self.cameraAdaption
    
    def updatePlayerPosition(self):
        # 在index的循环次数中，不进行碰撞检测，用来让玩家向下跳跃
        if self.index > 0:
            self.index -= 1
            self.player1.rect.x += self.player1.xSpeed
            self.player1.rect.y += self.player1.ySpeed
            self.player1.isDown = False
        else:
            # 首先更新y的位置
            self.player1.rect.y += self.player1.ySpeed
            # 玩家向下跳跃，35次循环内不进行碰撞检测
            if self.player1.state == State.JUMP and self.player1.isDown:
                self.index = 35
            # 玩家向上跳跃，15次循环内不进行碰撞检测
            elif self.player1.state == State.JUMP and self.player1.isUp:
                self.index = 15
            else:
                # 检测碰撞
                # 这里是玩家和所有碰撞组中的碰撞体检测碰撞，如果发生了碰撞，就会返回碰撞到的碰撞体对象
                collider = pygame.sprite.spritecollideany(self.player1, MainGame.playerColliderGroup)
                # 如果发生碰撞，判断是不是在河里
                if collider in MainGame.playerRiverGroup:
                    self.riverCollide()
                # 判断是不是在陆地上
                elif collider in MainGame.playerLandGroup:
                    self.player1.isInWater = False
                # 如果发生碰撞
                if collider:
                    if MainGame.player1.isInvincible:
                        # 玩家落地不无敌
                        MainGame.player1.isInvincible = False
                    # 判断一下人物的y速度，如果大于0，则说明玩家已经接触到了碰撞体表面，需要让玩家站在表面，不掉下去
                    if self.player1.ySpeed > 0:
                        self.player1.ySpeed = 0
                        self.player1.state = State.WALK
                        self.player1.rect.bottom = collider.rect.top
                else:
                    # 否则的话，我们创建一个玩家的复制
                    tempPlayer = copy.copy(self.player1)
                    # 让玩家的纵坐标—+1，看看有没有发生碰撞
                    tempPlayer.rect.y += 1
                    # 如果没有发生碰撞，就说明玩家下面不是碰撞体，是空的
                    if not pygame.sprite.spritecollideany(tempPlayer, MainGame.playerColliderGroup):
                        # 如果此时不是跳跃状态，那么就让玩家变成下落状态，因为玩家在跳跃时，是向上跳跃，不需要对下面的物体进行碰撞检测
                        if tempPlayer.state != State.JUMP:
                            self.player1.state = State.FALL
                    tempPlayer.rect.y -= 1
                    if tempPlayer.rect.y > 610:
                        if MainGame.player1.damage(1):
                            initPlayer1(MainGame.player1.life)
                
                # 与敌人碰撞
                if pygame.sprite.spritecollideany(MainGame.player1, MainGame.enemyGroup):
                    if MainGame.player1.damage(1):
                        MainGame.explodeList.append(Explode(MainGame.player1, ExplodeVariety.PLAYER1))
                        initPlayer1(MainGame.player1.life)
            
            # 更新x的位置
            self.player1.rect.x += self.player1.xSpeed
            # 同样的检查碰撞
            collider = pygame.sprite.spritecollideany(self.player1, MainGame.playerColliderGroup)
            # 如果发生了碰撞
            if collider:
                # 判断玩家的x方向速度，如果大于0，表示右边有碰撞体
                if self.player1.xSpeed > 0:
                    # 设置玩家的右边等于碰撞体的左边
                    self.player1.rect.right = collider.rect.left
                else:
                    # 左边有碰撞体
                    self.player1.rect.left = collider.rect.right
                self.player1.xSpeed = 0
            
            tempPlayer = copy.copy(self.player1)
            tempPlayer.rect.y += 1
            if c := pygame.sprite.spritecollideany(tempPlayer, MainGame.playerColliderGroup):
                if c in MainGame.playerLandGroup:
                    self.player1.isInWater = False
                elif c in MainGame.playerRiverGroup:
                    self.player1.isInWater = True
            
            # 玩家与桥碰撞的逻辑
            if c := pygame.sprite.spritecollideany(tempPlayer, MainGame.commonColliderGroup):
                MainGame.player1.isInWater = False
                # 玩家碰到桥
                if not self.bridgeExploding:
                    self.bridgeExploding = True
                    # 把碰到的桥放到列表里
                    self.hasCollidedBridge.append(c)
                    # MainGame.commonColliderGroup.remove(c)
            else:
                # 获取玩家中心
                center = MainGame.player1.getCenter()
                # 遍历桥列表，看看玩家中心当前在哪一个桥的范围内
                for bridge in MainGame.bridgeList:
                    if bridge.rect.x + bridge.rect.width * 2 / 3 < center[0] < bridge.rect.x + bridge.rect.width:
                        # 找到了，那么就让这个桥爆炸
                        self.bridgeExploding = True
                        # 删除碰撞体
                        for collider in MainGame.commonColliderGroup:
                            if collider.rect.x < center[0] < collider.rect.x + collider.rect.width:
                                MainGame.commonColliderGroup.remove(collider)
            
            tempPlayer.rect.y -= 1
    
    def riverCollide(self):
        # 在河里设置isInWater
        self.player1.isInWater = True
        # 设置玩家在河里不能跳跃
        self.player1.isJumping = False
        # 默认落下去是站在河里的
        self.player1.isStanding = True
        # 玩家方向不能向下
        self.player1.isDown = False
        # 根据玩家方向，加载落入河中的一瞬间的图片
        if self.player1.direction == Direction.RIGHT:
            self.player1.image = self.player1.rightInWaterImage
        else:
            self.player1.image = self.player1.leftInWaterImage
    
    def generateEnemy(self):
        if -1505 < self.backRect.x < -1500:
            if self.enemyBoolList[0]:
                self.enemyBoolList[0] = False
                generateEnemy1(MainGame.player1.rect.x + 600, POSITION_1, Direction.LEFT, pygame.time.get_ticks())
                generateEnemy1(MainGame.player1.rect.x - 360, POSITION_1, Direction.RIGHT, pygame.time.get_ticks())
        
        if -1705 < self.backRect.x < -1700:
            if self.enemyBoolList[1]:
                self.enemyBoolList[1] = False
                generateEnemy1(MainGame.player1.rect.x - 360, POSITION_1, Direction.RIGHT, pygame.time.get_ticks())
                generateEnemy1(MainGame.player1.rect.x - 400, POSITION_1, Direction.RIGHT,
                               pygame.time.get_ticks())
        
        if -2005 < self.backRect.x < -2000:
            if self.enemyBoolList[2]:
                self.enemyBoolList[2] = False
                generateEnemy2(MainGame.player1.rect.x + 540, 465)
    
    def initBackground(self):
        # 读取背景图片
        self.background = pygame.image.load('../Image/Map/1/Background/First(No Bridge).png')
        self.backRect = self.background.get_rect()
        self.background = pygame.transform.scale(
            self.background,
            (int(self.backRect.width * MAP_SCALE),
             int(self.backRect.height * MAP_SCALE))
        )
        self.backRect.x = -1280
    
    def drawLifeImage(self, window):
        # 如果玩家的生命值大于3，那么生命值图标就显示3个
        if MainGame.player1.life > 3:
            number = 3
        # 否则，有几个显示几个，肯定不超过三个
        else:
            number = MainGame.player1.life
        rect = self.lifeImage.get_rect()
        # 设置生命值图标的显示位置
        rect.y = 5
        for i in range(number):
            # 每个图标之间的距离为25像素
            rect.x = 5 + i * 20
            window.blit(self.lifeImage, rect)
    
    def bridgeExplode(self):
        if self.bridgeExploding:
            self.bridgeExploding = False
            if len(MainGame.bridgeList) > 0:
                # 取出第一个， 创建爆炸，放入爆炸列表
                bridge = MainGame.bridgeList[0]
                # 把该桥移除
                MainGame.bridgeList.remove(bridge)
                # 创建爆炸，指定爆炸类型，并且是按照时间来显示爆炸图片
                explode = Explode(bridge, ExplodeVariety.BRIDGE, True)
                # 设置时间
                explode.lastTime = pygame.time.get_ticks()
                MainGame.explodeList.append(explode)
                # 检查列表中是不是有元素
                if len(self.hasCollidedBridge) > 0:
                    # 把第一个元素取出来，并且删除这个元素，这里是碰撞的桥会被放到列表里，所以删除的是碰到的桥
                    c = self.hasCollidedBridge.pop()
                    MainGame.commonColliderGroup.remove(c)
    
    def initBoss(self):
        # boss枪管口1
        boss = BossPart(5920 * RATIO, 131 * MAP_SCALE, pygame.time.get_ticks())
        # boss = BossPart(320 * RATIO, 131 * MAP_SCALE, pygame.time.get_ticks())
        MainGame.enemyList.append(boss)
        MainGame.allSprites.add(boss)
        MainGame.enemyGroup.add(boss)
        self.boss.append(boss)
        # boss枪管口2
        boss = BossPart(5965 * RATIO, 131 * MAP_SCALE, pygame.time.get_ticks())
        # boss = BossPart(365 * RATIO, 131 * MAP_SCALE, pygame.time.get_ticks())
        MainGame.enemyList.append(boss)
        MainGame.allSprites.add(boss)
        MainGame.enemyGroup.add(boss)
        self.boss.append(boss)
        # boss弱点
        boss = BossPart(5950 * RATIO, 170 * MAP_SCALE, pygame.time.get_ticks(), 4)
        # boss = BossPart(350 * RATIO, 170 * MAP_SCALE, pygame.time.get_ticks(), 4)
        MainGame.enemyList.append(boss)
        MainGame.allSprites.add(boss)
        MainGame.enemyGroup.add(boss)
        self.boss.append(boss)
    
    def loadApproachAnimation(self):
        # 读取进场图片
        approach = pygame.image.load('../Image/Map/1/Background/First(Approach).png')
        approachRect = self.background.get_rect()
        approach = pygame.transform.scale(
            approach,
            (int(approachRect.width * 1),
             int(approachRect.height * 1))
        )
        approachRect.x = 0
        # 设置进场图片移动速度
        cameraAdaption = 3
        # 记录当前时间
        currentTime = 0
        # 创建一张黑色的图片，用来盖住选择图标
        image = pygame.Surface((50, 50)).convert()
        image.fill((0, 0, 0))
        # 记录是否播放音效，播放了就要画了
        isPlayed = False
        showTime = pygame.time.get_ticks()
        lastingTime = pygame.time.get_ticks()
        keys = ''
        while 1:
            MainGame.window.blit(approach, approachRect)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    # 上上下下左右左右BABA A是跳跃键，B是射击键
                    if event.key == pygame.K_w:
                        keys += 'w'
                    elif event.key == pygame.K_s:
                        keys += 's'
                    elif event.key == pygame.K_j:
                        keys += 'b'
                    elif event.key == pygame.K_k:
                        keys += 'a'
                    elif event.key == pygame.K_RETURN:
                        if not isPlayed:
                            approachRect.x = -1435
                            Sound('../Sound/start.mp3').play()
                            currentTime = time.time()
                            isPlayed = True
            
            # 让背景向右走这么多距离
            if approachRect.x > -1435:
                approachRect.x -= cameraAdaption
            
            if isPlayed:
                # 设置图标一闪一闪的
                if abs(lastingTime - pygame.time.get_ticks()) > 400:
                    if 1200 > abs(showTime - pygame.time.get_ticks()) > 0:
                        MainGame.window.blit(image, (190, 390))
                    else:
                        showTime = pygame.time.get_ticks()
                        lastingTime = pygame.time.get_ticks()
            
            # 更新窗口
            pygame.display.update()
            # 设置帧率
            self.clock.tick(self.fps)
            fps = self.clock.get_fps()
            caption = '魂斗罗 - {:.2f}'.format(fps)
            pygame.display.set_caption(caption)
            
            # 如果时间超过60，就开始加载第一关
            if 100 > abs(time.time() - currentTime) * 10 > 60:
                print(keys)
                if keys == 'wwssbaba':
                    initPlayer1(30)
                break
    
    def runGame(self):
        self.loadApproachAnimation()
        self.run()

if __name__ == '__main__':
    MainGame().runGame()
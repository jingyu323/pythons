import pygame


class Explode:
    def __init__(self, object, size):
        # 获取爆炸对象的位置
        self.rect = object.rect
        self.images = []
        self.images.append([
            pygame.image.load('../tank/Image/Explode/Explode50x50.png'),
            pygame.image.load('../tank/Image/Explode/Explode50x50.png'),
            pygame.image.load('../tank/Image/Explode/Explode50x50.png'),
            pygame.image.load('../tank/Image/Explode/Explode50x50.png'),
            pygame.image.load('../tank/Image/Explode/Explode50x50.png')
        ])
        self.images.append([
            pygame.image.load('../tank/Image/Explode/Explode25x25.png'),
            pygame.image.load('../tank/Image/Explode/Explode25x25.png'),
            pygame.image.load('../tank/Image/Explode/Explode25x25.png'),
            pygame.image.load('../tank/Image/Explode/Explode25x25.png'),
            pygame.image.load('../tank/Image/Explode/Explode25x25.png')
        ])
        self.mode = 0
        if size == 25:
            self.mode = 1
        self.index = 0
        self.image = self.images[self.mode][self.index]
        self.isDestroy = False

    def draw(self, window):
        # 根据索引获取爆炸对象, 添加到主窗口
        # 让图像加载五次，这里可以换成五张大小不一样的爆炸图片，可以实现让爆炸效果从小变大的效果


        if self.index < len(self.images):
            self.image = self.images[self.mode][self.index]
            self.index += 1
            window.blit(self.image, self.rect)
        else:
            self.isDestroy = True
            self.index = 0

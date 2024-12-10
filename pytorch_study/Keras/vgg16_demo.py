from keras.src.applications.vgg16 import VGG16
from keras_preprocessing.image import ImageDataGenerator


def demo1():
    model = VGG16(weights='imagenet', include_top=True)
    model.summary()

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    batch_size = 32

    train_step = int( (2000+batch_size-1)/batch_size)*10
    test_step = int( (1000+batch_size-1)/batch_size)*10

    batch_generator = datagen.flow_from_directory(
        'data/train',
        target_size=(150,150),
        batch_size=batch_size,
        class_mode=None,  # 不生成标签 Node  categorical 分类标签
        shuffle=False # 不随机打乱
        #subset='training'
    )



if __name__ == '__main__':
    demo1();
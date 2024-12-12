from keras import Sequential
from keras.src.applications.vgg16 import VGG16
from keras.src.layers import Flatten, Dense, Dropout
from keras_preprocessing.image import ImageDataGenerator

"""
训练方法1：bottleneck

训练方法2：Finetune

"""

def demo1_bottleneck():
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


def demo_Finetune():
    vgg16_model = VGG16(weights='imagenet', include_top=False,input_shape=(150,150,3))
    vgg16_model.summary()
    top_mmodel = Sequential()
    top_mmodel.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
    top_mmodel.add(Dense(256, activation='relu'))
    top_mmodel.add(Dropout(0.5))
    top_mmodel.add(Dense(2, activation='softmax'))

    


if __name__ == '__main__':
    demo_Finetune();
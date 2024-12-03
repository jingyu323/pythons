from keras.src.legacy import layers

img_input = layers.Input(shape=input_shape)
x = layers.Conv2D(64, (3, 3), activation='relu')(img_input)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

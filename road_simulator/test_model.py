from roadsimulator.models.utils import get_datasets

train_X, train_Y, val_X, val_Y, _, _ = get_datasets('my_dataset', n_images=100)


from keras.layers import Input, Convolution2D, MaxPooling2D, Activation
from keras.layers import Flatten, Dense
from keras.models import Model

img_in = Input(shape=(70, 250, 3), name='img_in')
x = img_in

x = Convolution2D(1, 3, 3, activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)

x = Convolution2D(2, 3, 3, activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)

x = Convolution2D(2, 3, 3, activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)

x = Convolution2D(4, 3, 3, activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)

flat = Flatten()(x)

x = Dense(20)(flat)
x = Activation('relu')(x)

x = Dense(5)(x)
angle_out = Activation('softmax')(x)

model = Model(input=[img_in], output=[angle_out])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit([train_X], train_Y, batch_size=100, nb_epoch=20, validation_data=([val_X], val_Y))

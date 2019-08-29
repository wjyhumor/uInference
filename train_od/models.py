
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import *


def model_1(input_height, input_width, input_channel, max_box_per_image, nb_box, nb_class):
    input_image = Input(
        shape=(input_height, input_width, input_channel))
    true_boxes = Input(shape=(1, 1, 1, max_box_per_image, 4))

    model = Sequential()
    # Layer 1
    model.add(Conv2D(4, (3, 3), strides=1, padding='same', kernel_initializer=keras.initializers.glorot_normal(
        seed=None), input_shape=(input_height, input_width, input_channel)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2 - 5
    for i in range(0, 4):
        model.add(Conv2D(4*(2**i), (3, 3), strides=1, padding='same', kernel_initializer=keras.initializers.glorot_normal(
            seed=None)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 6
    model.add(Conv2D(64, (3, 3), strides=1, padding='same', kernel_initializer=keras.initializers.glorot_normal(
        seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 7
    model.add(Conv2D(64, (2, 2), strides=1, padding='same', kernel_initializer=keras.initializers.glorot_normal(
        seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    h, w = model.get_output_shape_at(-1)[1:3]

    # yolo2
    model.add(Conv2D(nb_box * (4 + 1 + nb_class), (1, 1),
                     strides=1, padding='same', name='DetectionLayer',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(Reshape((h, w, nb_box,
                      4 + 1 + nb_class)))
    #model.add(Lambda(lambda args: args[0])([model, true_boxes]))

    model = keras.models([input_image, true_boxes], model)

    # initialize the weights of the detection layer
    layer = model.layers[-4]
    weights = layer.get_weights()

    new_kernel = np.random.normal(
        size=weights[0].shape)/(h*w)
    new_bias = np.random.normal(
        size=weights[1].shape)/(h*w)

    layer.set_weights([new_kernel, new_bias])

    model.summary()
    return model

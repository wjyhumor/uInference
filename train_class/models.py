
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import *
from keras.applications.mobilenet import MobileNet
import load_data

num_classes = 10

def model_0():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    return model

# stm32 use
def model_1():
    model = Sequential()
    #model.add(Reshape((input_pixels, input_pixels, 1), input_shape=(input_pixels, input_pixels)))
    model.add(Conv2D(8, (5, 5), strides=1, padding='same', kernel_initializer=keras.initializers.glorot_normal(
        seed=None), input_shape=(load_data.resize_w, load_data.resize_h, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    # model.add(Dropout(0.5))
    model.add(Dense(
        num_classes, kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(Activation('softmax'))
    print(model.summary())
    return model

def model_2():
    model = Sequential()
    model.add(Conv2D(8, (3, 3), strides=1, padding='same', kernel_initializer=keras.initializers.glorot_normal(
        seed=None), input_shape=(load_data.resize_w, load_data.resize_h, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3), strides=1, padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    # model.add(Dropout(0.5))
    model.add(Dense(
        num_classes, kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(Activation('softmax'))
    print(model.summary())
    return model


def model_3():
    model = Sequential()
    model.add(Conv2D(8, (3, 3), strides=1, padding='same', kernel_initializer=keras.initializers.glorot_normal(
        seed=None), input_shape=(load_data.resize_w, load_data.resize_h, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3), strides=1, padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    # model.add(Dropout(0.5))
    model.add(Dense(
        num_classes, kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(Activation('softmax'))
    print(model.summary())
    return model

def model_4():
    model = Sequential()
    model.add(Conv2D(8, (3, 3), strides=1, padding='same', kernel_initializer=keras.initializers.glorot_normal(
        seed=None), input_shape=(load_data.resize_w, load_data.resize_h, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3), strides=1, padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3), strides=1, padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3), strides=1, padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    #model.add(Dropout(0.5))
    model.add(Dense(
        num_classes, kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(Activation('softmax'))
    print(model.summary())
    return model


def model_5():
    model = Sequential()
    model.add(Conv2D(8, (3, 3), strides=1, padding='same', kernel_initializer=keras.initializers.glorot_normal(
        seed=None), input_shape=(load_data.resize_w, load_data.resize_h, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3), strides=1, padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    # model.add(Dropout(0.5))
    model.add(Dense(
        num_classes, kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(Activation('softmax'))
    print(model.summary())
    return model

def model_6():
    model = Sequential()
    model.add(Conv2D(8, (3, 3), strides=1, padding='same', kernel_initializer=keras.initializers.glorot_normal(
        seed=None), input_shape=(load_data.resize_w, load_data.resize_h, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3), strides=1, padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # model.add(Dropout(0.5))
    model.add(Dense(
        num_classes, kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(Activation('softmax'))
    print(model.summary())
    return model

def model_7():
    model = Sequential()
    model.add(Conv2D(8, (3, 3), strides=1, padding='same', kernel_initializer=keras.initializers.glorot_normal(
        seed=None), input_shape=(load_data.resize_w, load_data.resize_h, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3), strides=1, padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3), strides=1, padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3), strides=1, padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    #model.add(Dropout(0.5))
    model.add(Dense(
        num_classes, kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(Activation('softmax'))
    print(model.summary())
    return model

def model_8():
    model = Sequential()
    model.add(Conv2D(8, (3, 3), strides=1, padding='same', kernel_initializer=keras.initializers.glorot_normal(
        seed=None), input_shape=(load_data.resize_w, load_data.resize_h, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3), strides=1, padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3), strides=1, padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3), strides=1, padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3), strides=1, padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3), strides=1, padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    #model.add(Dropout(0.5))
    model.add(Dense(
        num_classes, kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(Activation('softmax'))
    print(model.summary())
    return model


def model_9():
    model = Sequential()
    model.add(Conv2D(8, (3, 3), strides=1, padding='same', kernel_initializer=keras.initializers.glorot_normal(
        seed=None), input_shape=(load_data.resize_w, load_data.resize_h, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3), strides=1, padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3), strides=1, padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3), strides=1, padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3), strides=1, padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3), strides=1, padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3), strides=1, padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3), strides=1, padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same',
                     kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    #model.add(Dropout(0.5))
    model.add(Dense(
        num_classes, kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(Activation('softmax'))
    print(model.summary())
    return model

# server use
def model_server():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu",
                     input_shape=(load_data.resize_w, load_data.resize_h, 1)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    #model.add(Conv2D(filters=256, kernel_size = (3,3), activation="relu"))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    # model.add(BatchNormalization())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    print(model.summary())
    return model

def model_mobilenet():
    model = Sequential()
    model.add(MobileNet(input_shape=(load_data.resize_w, load_data.resize_h, 1), include_top=False, weights=None), kernel_initializer=keras.initializers.glorot_normal(seed=None))

    model.add(Flatten())
    model.add(Dense(
        num_classes, kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(Activation('softmax'))
    print(model.summary())
    return model

import re
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
print(tf.__version__)
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.models import load_model

import models
import load_data
import save_model_builder

def train(model_index, file_train, file_test, load_model_flag, load_model_name, batch_size, epochs):
    train_images, train_labels = load_data.load_data_train(file_train)
    test_images, test_labels = load_data.load_data_test(file_test)
    train_images = train_images.reshape(-1, load_data.resize_w, load_data.resize_h, 1)
    test_images = test_images.reshape(-1, load_data.resize_w, load_data.resize_h, 1)
    print("Image shape:" + str(train_images.shape))
    print("Label shape:" + str(train_labels.shape))
    # Augmentation---------------------------------------------
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=20,
        zoom_range=0.3,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)
    train_gen = datagen.flow(train_images, train_labels, batch_size=batch_size)
    #test_gen = datagen.flow(test_images, test_labels, batch_size=batch_size)

    # build model------------------------------
    if load_model_flag == False:
        if model_index == 1:
            model = models.model_1()
        elif model_index == 3:
            model = models.model_3()
    else:
        model = load_model(str(load_model_name))
        print(model.summary())
    # set optimizer----------------------------------------
    optimizer = Adam(lr=1e-3)
    model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',  # if the labels are integers
                metrics=['accuracy'])

    # save checkpoint--------------------------------------
    filepath = "weights-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(
        str(filepath), monitor='val_acc', verbose=1, save_best_only=False, period=10)
    # Tensorboard: tensorboard --logdir=./logs
    callbacks = [checkpoint,
                keras.callbacks.TensorBoard(log_dir='./logs',
                                            histogram_freq=0,
                                            batch_size=batch_size,
                                            write_graph=True,
                                            write_grads=False,
                                            write_images=False,
                                            embeddings_freq=0,
                                            embeddings_layer_names=None,
                                            embeddings_metadata=None,
                                            embeddings_data=None, 
                                            update_freq='epoch')]
    # train-----------------------------------------------
    # history = model.fit(train_images, train_labels, validation_split=0.33,
    #                    batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint])
    history = model.fit_generator(train_gen,
                                steps_per_epoch=len(train_images) // batch_size,
                                validation_data=(test_images, test_labels),
                                validation_steps=len(test_images) // batch_size,
                                epochs=epochs, verbose=1, callbacks=callbacks)

    # test-----------------------------------------------
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("Test accuracy:"+str(test_acc))

    # Plot training & validation loss values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model acc')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def load_save_model(load_model_name):
    model = load_model(str(load_model_name))
    print(model.summary())
    model_json = model.to_json()
    with open("save_model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("save_weights.h5")
    print("Saved model to disk")

if __name__ == '__main__':
    model_index = 3 # 1: model for edge; 3: model for cloud 
    load_model_flag = False # if trained from a weight, then True; else, false.
    batch_size = 16
    epochs = 50
    file_train = '/home/neusoft/amy/uInference/data/water_elec_0516_0625.digits_front.train'
    file_test = '/home/neusoft/amy/uInference/data/water_elec_0516_0625.digits_front.test'
    load_model_name = '/home/neusoft/amy/uInference/weights-model1.hdf5'
    train(model_index, file_train, file_test, load_model_flag, load_model_name, batch_size, epochs)
    
    save_model_name = '/home/neusoft/amy/uInference/weights-3-front.hdf5'
    save_model_builder.save_model_builder(save_model_name)


import save_model_builder
import load_data
import models
import argparse
import re
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
print("tensorflow version:" + tf.__version__)
from list_oper import WriteList
from list_oper import ShuffleAllList
from list_oper import SplitList
from test import test
from save_model_builder import save_model_builder

argparser = argparse.ArgumentParser(description='Train OCR model')

argparser.add_argument(
    '-type',
    type=int,
    help='type of model to train, 1: for edge unit, 3: for cloud')

argparser.add_argument(
    '-batch_size',
    type=int,
    help='batch size to train')

argparser.add_argument(
    '-epochs',
    type=int,
    help='epochs to train')

argparser.add_argument(
    '-pretrain',
    type=int,
    help='use pretrained model flag, 1: use pretrained model, 0: not use')

argparser.add_argument(
    '-pretrain_model',
    help='path for pretrained model(option)')

argparser.add_argument(
    '-reload_train',
    type=int,
    help='flag for reload training data or not, 1: reload, 0: do not reload')

argparser.add_argument(
    '-reload_test',
    type=int,
    help='flag for reload test data or not, 1: reload, 0: do not reload')

argparser.add_argument(
    '-data_path',
    help='path for the list of data')

argparser.add_argument(
    '-save_model',
    help='name for model to save')



def train(model_type=1, 
          train_list="",
          test_list="", 
          pretrain_flag=0,
          batch_size=16, 
          epochs=50, 
          save_model_name="",
          pretrained_model_name=None,
          reload_train=True,
          reload_test=True):
    train_images, train_labels = load_data.load_data_train(
        train_list, reload_train)
    test_images, test_labels = load_data.load_data_test(test_list, reload_test)
    train_images = train_images.reshape(-1,
                                        load_data.resize_w, load_data.resize_h, 1)
    test_images = test_images.reshape(-1,
                                      load_data.resize_w, load_data.resize_h, 1)
    #print("Image shape:" + str(train_images.shape))
    #print("Label shape:" + str(train_labels.shape))
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
    if pretrain_flag == 0:
        if model_type == 1:
            model = models.model_9()
        elif model_type == 3:
            model = models.model_server()
    else:
        model = load_model(str(pretrained_model_name))
        print(model.summary())
    # set optimizer----------------------------------------
    optimizer = Adam(lr=1e-3)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',  # if the labels are integers
                  metrics=['accuracy'])

    # save checkpoint--------------------------------------
    #save_model_name = "./tmp/weights-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(
        str(save_model_name), monitor='val_acc', verbose=1, save_best_only=True, period=epochs)
    # Tensorboard: tensorboard --logdir=./logs
    """
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
    """
    callbacks = [checkpoint]
    # train-----------------------------------------------
    # history = model.fit(train_images, train_labels, validation_split=0.33,
    #                    batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint])
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=len(
                                      train_images) // batch_size,
                                  validation_data=(test_images, test_labels),
                                  validation_steps=len(
                                      test_images) // batch_size,
                                  epochs=epochs, verbose=1, callbacks=callbacks)

    # test-----------------------------------------------
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("Test accuracy:"+str(test_acc))
    """
    # Plot training & validation loss values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model acc')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    """


def _main_(args):
    model_type = args.type
    pretrain_flag = args.pretrain
    data_path = args.data_path
    save_model_name = args.save_model
    batch_size = args.batch_size
    epochs = args.epochs

    print("===========Parameters==============")
    print("model_type: " + str(model_type))
    print("pretrain_flag: " + str(pretrain_flag))
    print("data_path: " + data_path)
    print("save_model_name: " + str(save_model_name))
    print("batch_size: " + str(batch_size))
    print("epochs: " + str(epochs))
    print("===================================")

    # pre-process of data
    save_path = "./tmp/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    all_list = "./tmp/images_base_all.list"
    train_list = "./tmp/images_base_train.list"
    test_list = "./tmp/images_base_test.list"
    WriteList(data_path, all_list)
    images_number = ShuffleAllList(all_list)
    print("Images_number:" + str(images_number))
    SplitList(all_list, train_list, test_list, 0.8)
    
    # train 
    if pretrain_flag:
        pretrained_model_name = args.pretrain_model
        train(model_type, train_list, test_list, pretrain_flag,
              batch_size, epochs, save_model_name, pretrained_model_name,
              reload_train=args.reload_train, reload_test=args.reload_test)
    else:
        train(model_type, train_list, test_list, pretrain_flag,
              batch_size, epochs, save_model_name,
              reload_train=args.reload_train, reload_test=args.reload_test)
    
    # save model builder
    if model_type == 3:
        save_model_builder(save_model_name, os.path.splitext(save_model_name)[0])
    
    # test
    # test(test_list, save_model_name, reload_test=args.reload_test)


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)

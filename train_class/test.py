
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import statistics

import tensorflow as tf
from keras.models import load_model

import models
import load_data

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def test(file_test, load_model_name):
    # load data
    test_images, test_labels = load_data.load_data_test(file_test)
    test_images = test_images.reshape(-1,
                                      load_data.resize_w, load_data.resize_h, 1)

    # load model and predict
    model = load_model(str(load_model_name))
    predictions = model.predict(test_images)

    max_predictions = []
    for pre in predictions:
        max_predictions.append(float(np.max(pre)))
    #print(max_predictions)
    print(statistics.mean(max_predictions))
    plt.hist(max_predictions, bins=10)
    plt.show()

    # evaluate
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("Test accuracy:"+str(test_acc))

    # plot
    test_images = test_images.reshape(-1,
                                      load_data.resize_h, load_data.resize_w)
    #plot_error(test_images, test_labels, predictions)
    #plot_result(test_images, test_labels, predictions)


def plot_result(test_images, test_labels, predictions):
    # Color correct predictions in blue, incorrect predictions in red
    num_rows = 5
    num_cols = 5
    num_images = num_rows * num_cols
    # draw randomly the images
    while 1:
        plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
            index = random.randint(0, len(test_images) - 1)
            plot_image(index, predictions, test_labels, test_images)
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
            plot_value(index, predictions, test_labels)
        plt.show()


def plot_error(test_images, test_labels, predictions):
    # save predicted wrong images
    wrong_images = []
    wrong_predict = []
    wrong_index = []
    true_labels = []
    for i in range(len(test_labels)):
        if np.argmax(predictions[i]) != test_labels[i]:
            wrong_images.append(test_images[i])
            wrong_predict.append(predictions[i])
            wrong_index.append(i)
            true_labels.append(test_labels[i])

    print("error images' number:" + str(len(wrong_images)))
    if len(wrong_images) != 0:
        wrong_images = np.asarray(wrong_images, dtype=np.float)
        wrong_predict = np.asarray(wrong_predict, dtype=np.float)
        wrong_index = np.asarray(wrong_index, dtype=np.int)
        true_labels = np.asarray(true_labels, dtype=np.int)
        """
        for w in wrong_index:
            print w,
            print ",",
        print " "
        """
        # Color correct predictions in blue, incorrect predictions in red
        num_rows = 5
        num_cols = 5
        num_images = num_rows * num_cols
        index = 0
        while 1:
            plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
            for i in range(num_images):
                plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
                plot_wrong_image(index, wrong_predict, true_labels, wrong_images, wrong_index)
                plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
                plot_value(index, wrong_predict, true_labels)
                index = index + 1
                if index >= len(true_labels):
                    break
            plt.show()
            if index >= len(true_labels):
                break

def plot_image(i, predictions, true_label, img):
    pred, true_label, img = predictions[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(pred)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{}: {} {:2.0f}% ({})".format(i,
                                             class_names[predicted_label],
                                             100 * np.max(pred),
                                             class_names[true_label]), 
                                             color=color)

def plot_wrong_image(i, predictions, true_label, img, wrong_index):
    pred, true_label, img = predictions[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(pred)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{}: {} {:2.0f}% ({})".format(wrong_index[i],
                                             class_names[predicted_label],
                                             100 * np.max(pred),
                                             class_names[true_label]), 
                                             color=color)

def plot_value(i, predictions, true_label):
    pred, true_label = predictions[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), pred, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(pred)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

if __name__ == '__main__':
    file_test = '/home/neusoft/amy/AT-201/data/water_elec_0516_0625.digits_end.test'#beilu_0820_blank.all'#
    load_model_name = '/home/neusoft/amy/AT-201/cubeai_train/weights-3-end.hdf5'

    test(file_test, load_model_name)

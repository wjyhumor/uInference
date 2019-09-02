#! /usr/bin/env python

import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from keras import backend as K
from frontend import YOLO
import utils


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.c = c
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        return self.score


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)
    if np.min(x) < t:
        x = x/np.min(x)*t
    e_x = np.exp(x)
    return e_x / e_x.sum(axis, keepdims=True)


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap(
        [box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap(
        [box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    union = w1*h1 + w2*h2 - intersect
    return float(intersect) / union


def decode_netout(netout, anchors, nb_class, obj_threshold=0.3, nms_threshold=0.3):
    grid_h, grid_w, nb_box = netout.shape[:3]
    boxes = []
    # decode the output by the network
    netout[..., 4] = _sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * \
        _softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold
    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row, col, b, 5:]
                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row, col, b, :4]
                    # center position, unit: image width
                    x = (col + _sigmoid(x)) / grid_w
                    # center position, unit: image height
                    y = (row + _sigmoid(y)) / grid_h
                    w = anchors[2 * b + 0] * \
                        np.exp(w) / grid_w  # unit: image width
                    h = anchors[2 * b + 1] * \
                        np.exp(h) / grid_h  # unit: image height
                    confidence = netout[row, col, b, 4]
                    box = BoundBox(x-w/2, y-h/2, x+w/2, y +
                                   h/2, confidence, classes)
                    boxes.append(box)
    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(
            reversed(np.argsort([box.classes[c] for box in boxes])))
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0:
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0
    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]
    return boxes

def save_layer(layer_to_save):
    path = "../temp_save_layer_predict.txt"
    fout = open(path, 'w')
    """
    for i in range(layer_to_save.shape[1]):
        for j in range(layer_to_save.shape[2]):
            fout.write('{:=9f},'.format(layer_to_save[0, i, j, 0]))
        fout.write('\n')
    return 
    """
    if len(layer_to_save.shape) == 4:
        for l in range(layer_to_save.shape[3]): # output
            for j in range(layer_to_save.shape[1]): # height
                for k in range(layer_to_save.shape[2]): # width
                    for i in range(layer_to_save.shape[0]): # input
                        fout.write('{:=9f},'.format(layer_to_save[i, j, k, l]))
                fout.write('\n')
    elif len(layer_to_save.shape) == 3:
        for j in range(layer_to_save.shape[1]):
            for k in range(layer_to_save.shape[2]):
                for i in range(layer_to_save.shape[0]):
                    fout.write('{:=9f},'.format(layer_to_save[i, j, k]))
                fout.write('\n')

def _main_():
    image_path = "../ex_od.jpg"
    saved_config_name = "../models_od/tiny_yolo_ocr_6.json"
    saved_weights_name = "../models_od/tiny_yolo_ocr_6.h5"
    # model
    yolo = YOLO(backend="Tiny Yolo_5",
                input_width=320,
                input_height=105,
                input_channel=1,
                labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                max_box_per_image=10,
                anchors=[1.05, 5.13, 1.10, 3.60, 1.16,
                         5.14, 1.24, 4.22, 1.34, 5.24],
                saved_config_name=saved_config_name)
    # weights
    yolo.load_weights(saved_weights_name)
    # input image
    image_org = cv2.imread(image_path)
    image_h, image_w, image_c = image_org.shape
    print(image_org.shape)
    image = cv2.resize(image_org, (yolo.input_width, yolo.input_height))
    cv2.imwrite(image_path,image[:, :, 1])
    image = yolo.feature_extractor.normalize(image)
    #image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    image = np.reshape(image[:, :, 1], (image.shape[0], image.shape[1], 1))
    input_image = image[:, :, ::-1]
    input_image = np.expand_dims(input_image, 0)

    dummy_array = np.zeros((1, 1, 1, 1, yolo.max_box_per_image, 4))
    # forward NN
    netout = yolo.model.predict([input_image, dummy_array])[0]
    # get box
    boxes = decode_netout(netout, yolo.anchors, yolo.nb_class)
    print(len(boxes), 'boxes are found')
    # draw box
    image_show = utils.draw_boxes(
        image_org, boxes, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image_show)

    print"=================================="
    print(np.shape(image))
    print("input shape:"),
    print(np.shape(input_image))
    get_layer_output = K.function([yolo.model.layers[0].input],
                                  [yolo.model.layers[1].get_output_at(1)])
    layer_output = get_layer_output([input_image, dummy_array])[0]
    print "layer_output1:", np.shape(layer_output)
    # print layer_output

    """
    weights = yolo.model.layers[1].get_weights()
    print(len(weights))
    for i in range(len(weights)):
        w = yolo.model.layers[1].get_weights()[i]
        print(w.shape)
    """
    print"----------------"
    get_layer_output = K.function([yolo.model.layers[0].input],
                                  [yolo.model.layers[2].output])
    layer_output = get_layer_output([input_image, dummy_array])[0]
    w = yolo.model.layers[2].get_weights()[0]
    b = yolo.model.layers[2].get_weights()[1]
    print "layer_output2:", np.shape(layer_output)
    print "w:", w.shape, "b:", b.shape

    print"----------------"
    get_layer_output = K.function([yolo.model.layers[0].input],
                                  [yolo.model.layers[3].output])
    layer_output = get_layer_output([input_image, dummy_array])[0]
    print "layer_output3:", np.shape(layer_output)

    print"----------------"
    layer_number = 24
    get_layer_output = K.function([yolo.feature_extractor.feature_extractor.layers[0].input],
                                  [yolo.feature_extractor.feature_extractor.layers[layer_number].output])
    layer_output = get_layer_output([input_image])[0]
    print yolo.feature_extractor.feature_extractor.layers[layer_number].name, "'s output:", np.shape(layer_output)
    save_layer(layer_output)


if __name__ == '__main__':
    _main_()

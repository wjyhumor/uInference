#! /usr/bin/env python

import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
from utils import decode_netout, compute_overlap, compute_ap
from keras import backend as K

def _main_():
    image_path   = "../ex_od.jpg"
    saved_config_name  = "../models_od/tiny_yolo_ocr_6.json"
    saved_weights_name = "../models_od/tiny_yolo_ocr_6.h5"

    yolo = YOLO(backend             = "Tiny Yolo_5",
                input_width         = 320, 
                input_height        = 105, 
                input_channel       = 1, 
                labels              = ["0","1","2","3","4","5","6","7","8","9"], 
                max_box_per_image   = 10,
                anchors             = [1.05,5.13, 1.10,3.60, 1.16,5.14, 1.24,4.22, 1.34,5.24],
                saved_config_name   = saved_config_name)

    yolo.load_weights(saved_weights_name)

    image_org = cv2.imread(image_path)
    image_h, image_w, _ = image_org.shape

    image = cv2.resize(image_org, (yolo.input_width, yolo.input_height))
    image = yolo.feature_extractor.normalize(image)
    image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    image = np.reshape(image, (image.shape[0], image.shape[1], 1))

    input_image = image[:, :, ::-1]
    input_image = np.expand_dims(input_image, 0)
    dummy_array = np.zeros((1, 1, 1, 1, yolo.max_box_per_image, 4))

    netout = yolo.model.predict([input_image, dummy_array])[0]
    boxes = decode_netout(netout, yolo.anchors, yolo.nb_class)
    print(len(boxes), 'boxes are found')
    image_show = draw_boxes(image_org, boxes, ["0","1","2","3","4","5","6","7","8","9"])
    cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image_show)

    print"----------------"
    print(np.shape(image))
    print(np.shape(input_image))
    get_layer_output = K.function([yolo.model.layers[0].input],
                              [yolo.model.layers[1].get_output_at(1)])
    layer_output = get_layer_output([input_image, dummy_array])[0]
    print "layer_output1:", np.shape(layer_output)
    
    weights = yolo.model.layers[1].get_weights()
    print(len(weights))
    for i in range(len(weights)):
        w = yolo.model.layers[1].get_weights()[i]
        print(w.shape)

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
    get_layer_output = K.function([yolo.model.layers[0].input],
                              [yolo.model.layers[5].output])
    layer_output = get_layer_output([input_image, dummy_array])[0]
    print "layer_output5:", np.shape(layer_output)

if __name__ == '__main__':
    _main_()

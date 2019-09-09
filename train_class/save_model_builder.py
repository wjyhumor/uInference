import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import random
import subprocess                                                                                                        
import tensorflow as tf
import keras
from keras import backend as K                        
from keras.models import load_model
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image                                                                                     

argparser = argparse.ArgumentParser(description='Save model builder for tensorflow builder')

argparser.add_argument(
    '-i',
    '--input',
    help='the model for convert')

argparser.add_argument(
    '-o',
    '--output',
    help='the name to save')


def save_model_builder(model_name, save_name):     
    if os.path.exists(save_name) & os.path.isdir(save_name):
        subprocess.call("rm -rf " + save_name, shell=True)
    model = load_model(str(model_name))
    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'image': model.input}, outputs={'scores': model.output})       
    builder = tf.saved_model.builder.SavedModelBuilder(save_name)         
    builder.add_meta_graph_and_variables(                          
        sess=K.get_session(),                                          
        tags=[tf.saved_model.tag_constants.SERVING],          
        signature_def_map={                                            
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:signature
        })                                                            
    builder.save()
    print("Model builder saved to " + save_name)
       

if __name__ == '__main__':
    args = argparser.parse_args()
    save_model_builder(args.input, args.output)

# This model can be loaded in other langauges using the C API:                                                                               
# TF_SessionOptions* opts = TF_NewSessionOptions();                                                                                          
# const char* tags[] = {"serve"};  // tf.saved_model.tag_constants.SERVING                                                                   
# TF_LoadSessionFromSavedModel(opts, NULL, "saved_model", tags, 1, graph, NULL, status);
#
# This is what is used by the:
# - Java API: https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/SavedModelBundle
# - Go API: https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go#LoadSavedModel
# etc.
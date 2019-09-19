# convert Keras model to binary file
from keras.models import model_from_json
from keras.models import Sequential
from keras.models import load_model
import json
import struct

layer_dic = {"End":0, "Conv2D": 1, "BatchNormalization": 2, "Activation": 3, "MaxPooling2D": 4,
             "Flatten": 5, "Dense": 6, "InputLayer": 7, "Model": 8, "Reshape": 9,
             "Lambda": 10, "LeakyReLU": 11, "DepthwiseConv2D": 12, "ZeroPadding2D": 13,
             "ReLU": 14}
padding_dic = {"valid": 1, "same": 2}
activation_dic = {"relu": 1, "softmax": 2}

# load single-file model and save model as separate files: json & weights
def load_save_model(load_model_name, save_model_name, save_weights_name):
    model = load_model(str(load_model_name))
    print(model.summary())
    model_json = model.to_json()
    with open(save_model_name, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(save_weights_name)

# save model weights as self-defined format binary/txt file
def save_weights(save_type, json_name="save_model.json", weights="save_model.h5", output="save_model.dat"):
    arch = open(json_name).read()
    model = model_from_json(arch)
    model.load_weights(weights)
    print(model.summary())
    arch = json.loads(arch)
    if save_type == 'txt':
        open_type = 'w'
    elif save_type == 'binary':
        open_type = 'wb'
    with open(output, open_type) as fout:
        for ind, l in enumerate(arch["config"]["layers"]):
            print('layer ' + str(ind) + ' ' + l['class_name'])
            if save_type == 'txt':
                fout.write(l['class_name'] + '\n')
            elif save_type == 'binary':
                fout.write(struct.pack('>B', layer_dic[l['class_name']]))

            if l['class_name'] == 'Conv2D':
                W = model.layers[ind].get_weights()[0]
                if l['config']['use_bias'] == True:
                    b = model.layers[ind].get_weights()[1]
                if save_type == 'txt':
                    fout.write(str(W.shape[0]) + 
                               ' ' + str(W.shape[1]) +
                               ' ' + str(W.shape[2]) + 
                               ' ' + str(W.shape[3]) +
                               ' ' + str(l['config']['strides'][0]) +
                               ' ' + str(l['config']['strides'][1]) +
                               ' ' + str(l['config']['padding']) +
                               ' ' + str(l['config']['use_bias']) + '\n')
                    for o in range(W.shape[3]):
                        for k in range(W.shape[2]):
                            for i in range(W.shape[0]):
                                for j in range(W.shape[1]):
                                    fout.write(str(W[i, j, k, o]) + ',')
                            fout.write('\n')
                        if l['config']['use_bias'] == True:
                            fout.write(str(b[o]) + '\n')
                elif save_type == 'binary':
                    fout.write(struct.pack('i', W.shape[0]))
                    fout.write(struct.pack('i', W.shape[1]))
                    fout.write(struct.pack('i', W.shape[2]))
                    fout.write(struct.pack('i', W.shape[3]))
                    fout.write(struct.pack('>B', l['config']['strides'][0]))
                    fout.write(struct.pack('>B', l['config']['strides'][1]))
                    fout.write(struct.pack(
                        '>B', padding_dic[l['config']['padding']]))
                    if l['config']['use_bias']:
                        fout.write(struct.pack('>B', 1))
                    else:
                        fout.write(struct.pack('>B', 0))
                    for o in range(W.shape[3]): #output
                        for k in range(W.shape[2]): #input 
                            for i in range(W.shape[0]): #height
                                for j in range(W.shape[1]): #width
                                    fout.write(struct.pack('f', W[i, j, k, o]))
                        if l['config']['use_bias'] == True:
                            fout.write(struct.pack('f', b[o]))

            elif l['class_name'] == 'BatchNormalization':
                gamma = model.layers[ind].get_weights()[0]
                beta = model.layers[ind].get_weights()[1]
                mean = model.layers[ind].get_weights()[2]
                variance = model.layers[ind].get_weights()[3]
                if save_type == 'txt':
                    for i in range(0, len(gamma)):
                        fout.write(str(gamma[i]) + ',')
                        fout.write(str(beta[i]) + ',')
                        fout.write(str(mean[i]) + ',')
                        fout.write(str(variance[i]) + '\n')
                elif save_type == 'binary':
                    for i in range(0, len(gamma)):
                        fout.write(struct.pack('f', gamma[i]))
                        fout.write(struct.pack('f', beta[i]))
                        fout.write(struct.pack('f', mean[i]))
                        fout.write(struct.pack('f', variance[i]))

            elif l['class_name'] == 'Activation':
                if save_type == 'txt':
                    fout.write(l['config']['activation'] + '\n')
                elif save_type == 'binary':
                    fout.write(struct.pack('>B', activation_dic[l['config']['activation']]))

            elif l['class_name'] == 'LeakyReLU':
                if save_type == 'txt':
                    fout.write(str(l['config']['alpha']) + '\n')
                elif save_type == 'binary':
                    fout.write(struct.pack(
                        'f', l['config']['alpha']))

            elif l['class_name'] == 'MaxPooling2D':
                if save_type == 'txt':
                    fout.write(str(l['config']['pool_size'][0]) +
                           ' ' + str(l['config']['pool_size'][1]) +
                           ' ' + str(l['config']['padding']) + '\n')
                elif save_type == 'binary':
                    fout.write(struct.pack('>B', l['config']['pool_size'][0]))
                    fout.write(struct.pack('>B', l['config']['pool_size'][1]))
                    fout.write(struct.pack(
                        '>B', padding_dic[l['config']['padding']]))
                        
            elif l['class_name'] == 'Flatten':
                print(l['config']['name'])

            elif l['class_name'] == 'Dense':
                W = model.layers[ind].get_weights()[0]
                b = model.layers[ind].get_weights()[1]
                if save_type == 'txt':
                    fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + '\n')
                    for j in range(W.shape[1]):
                        for i in range(W.shape[0]):
                            fout.write(str(W[i, j]) + ',')
                        fout.write('\n')
                        fout.write(str(b[j]) + '\n')
                elif save_type == 'binary':
                    fout.write(struct.pack('>B', W.shape[0]))
                    fout.write(struct.pack('>B', W.shape[1]))
                    for j in range(W.shape[1]):
                        for i in range(W.shape[0]):
                            fout.write(struct.pack('f', W[i, j]))
                        fout.write(struct.pack('f', b[j]))
                        
            elif l['class_name'] == 'Model':
                sublayer_ind = 0
                for i in range(len(l['config']['layers'])):
                    name = l['config']['layers'][i]["class_name"]
                    print(name)
                    if save_type == 'txt':
                        fout.write(name + '\n')
                    elif save_type == 'binary':
                        fout.write(struct.pack('>B', layer_dic[name]))

                    if name == 'Conv2D':
                        W = model.layers[ind].get_weights()[sublayer_ind]
                        sublayer_ind += 1
                        if l['config']['layers'][i]["config"]["use_bias"] == True:
                            b = model.layers[ind].get_weights()[sublayer_ind]
                            sublayer_ind += 1
                        if save_type == 'txt':
                            fout.write(str(W.shape[0]) + 
                                   ' ' + str(W.shape[1]) +
                                   ' ' + str(W.shape[2]) + 
                                   ' ' + str(W.shape[3]) +
                                   ' ' + str(l['config']['layers'][i]['config']['strides'][0]) +
                                   ' ' + str(l['config']['layers'][i]['config']['strides'][1]) +
                                   ' ' + str(l['config']['layers'][i]['config']['padding']) +
                                   ' ' + str(l['config']['layers'][i]["config"]["use_bias"]) + '\n')
                            for o in range(W.shape[3]):
                                for k in range(W.shape[2]):
                                    for m in range(W.shape[0]):
                                        for n in range(W.shape[1]):
                                            fout.write(str(W[m, n, k, o]) + ',')
                                    fout.write('\n')
                                if l['config']['layers'][i]["config"]["use_bias"] == True:
                                    fout.write(str(b[o]) + '\n')
                        elif save_type == 'binary':
                            fout.write(struct.pack('i', W.shape[0]))
                            fout.write(struct.pack('i', W.shape[1]))
                            fout.write(struct.pack('i', W.shape[2]))
                            fout.write(struct.pack('i', W.shape[3]))
                            fout.write(struct.pack(
                                '>B', l['config']['layers'][i]['config']['strides'][0]))
                            fout.write(struct.pack(
                                '>B', l['config']['layers'][i]['config']['strides'][1]))
                            fout.write(struct.pack(
                                '>B', padding_dic[l['config']['layers'][i]['config']['padding']]))
                            if l['config']['layers'][i]['config']['use_bias']:
                                fout.write(struct.pack('>B', 1))
                            else:
                                fout.write(struct.pack('>B', 0))
                            #print W.shape[0], W.shape[1], W.shape[2], W.shape[3]
                            for o in range(W.shape[3]):
                                for k in range(W.shape[2]):
                                    for m in range(W.shape[0]):
                                        for n in range(W.shape[1]):
                                            fout.write(struct.pack(
                                                'f', W[m, n, k, o]))
                                if l['config']['layers'][i]["config"]["use_bias"] == True:
                                    fout.write(struct.pack('f', b[o]))
                    elif name == 'BatchNormalization':
                        gamma = model.layers[ind].get_weights()[sublayer_ind]
                        sublayer_ind += 1
                        beta = model.layers[ind].get_weights()[sublayer_ind]
                        sublayer_ind += 1
                        mean = model.layers[ind].get_weights()[sublayer_ind]
                        sublayer_ind += 1
                        variance = model.layers[ind].get_weights()[
                            sublayer_ind]
                        sublayer_ind += 1
                        if save_type == 'txt':
                            for k in range(0, len(gamma)):
                                fout.write(str(gamma[k]) + ',')
                                fout.write(str(beta[k]) + ',')
                                fout.write(str(mean[k]) + ',')
                                fout.write(str(variance[k]) + '\n')
                        elif save_type == 'binary':
                            for k in range(0, len(gamma)):
                                fout.write(struct.pack('f', gamma[k]))
                                fout.write(struct.pack('f', beta[k]))
                                fout.write(struct.pack('f', mean[k]))
                                fout.write(struct.pack('f', variance[k]))
                    elif name == 'LeakyReLU':
                        if save_type == 'txt':
                            fout.write(str(l['config']['layers'][i]
                                       ['config']['alpha']) + '\n')
                        elif save_type == 'binary':
                            fout.write(struct.pack(
                                'f', l['config']['layers'][i]['config']['alpha']))
                    elif name == 'ReLU':
                        if save_type == 'txt':
                            fout.write(str(l['config']['layers'][i]['config']['threshold']) + 
                                 ' ' + str(l['config']['layers'][i]['config']['max_value']) +
                                 ' ' + str(l['config']['layers'][i]['config']['negative_slope'])
                                        + '\n')
                        elif save_type == 'binary':
                            fout.write(struct.pack(
                                'f', l['config']['layers'][i]['config']['threshold']))
                            fout.write(struct.pack(
                                'f', l['config']['layers'][i]['config']['max_value']))
                            fout.write(struct.pack(
                                'f', l['config']['layers'][i]['config']['negative_slope']))
                    elif name == 'MaxPooling2D':
                        if save_type == 'txt':
                            fout.write(str(l['config']['layers'][i]['config']['pool_size'][0]) +
                                   ' ' + str(l['config']['layers'][i]['config']['pool_size'][1]) +
                                   ' ' + str(l['config']['layers'][i]['config']['padding']) + '\n')
                        elif save_type == 'binary':
                            fout.write(struct.pack(
                                '>B', l['config']['layers'][i]['config']['pool_size'][0]))
                            fout.write(struct.pack(
                                '>B', l['config']['layers'][i]['config']['pool_size'][1]))
                            fout.write(struct.pack(
                                '>B', padding_dic[l['config']['layers'][i]['config']['padding']]))
                    elif name == 'ZeroPadding2D':
                        if save_type == 'txt':
                            fout.write(str(l['config']['layers'][i]
                                       ['config']['padding']) + '\n')
                        elif save_type == 'binary':
                            fout.write(struct.pack(
                                '>B', l['config']['layers'][i]['config']['padding'][0][0])) # top_pad
                            fout.write(struct.pack(
                                '>B', l['config']['layers'][i]['config']['padding'][0][1])) # bottom_pad
                            fout.write(struct.pack(
                                '>B', l['config']['layers'][i]['config']['padding'][1][0])) # left_pad
                            fout.write(struct.pack(
                                '>B', l['config']['layers'][i]['config']['padding'][1][1])) # right_pad
                    elif name == 'DepthwiseConv2D':
                        W = model.layers[ind].get_weights()[sublayer_ind]
                        sublayer_ind += 1
                        if l['config']['layers'][i]["config"]["use_bias"] == True:
                            b = model.layers[ind].get_weights()[sublayer_ind]
                            sublayer_ind += 1
                        if save_type == 'txt':
                            fout.write(str(W.shape[0]) + 
                                   ' ' + str(W.shape[1]) +
                                   ' ' + str(W.shape[2]) + 
                                   ' ' + str(W.shape[3]) +
                                   ' ' + str(l['config']['layers'][i]['config']['kernel_size'][0]) +
                                   ' ' + str(l['config']['layers'][i]['config']['kernel_size'][1]) +
                                   ' ' + str(l['config']['layers'][i]['config']['strides'][0]) +
                                   ' ' + str(l['config']['layers'][i]['config']['strides'][1]) +
                                   ' ' + str(l['config']['layers'][i]['config']['padding']) +
                                   ' ' + str(l['config']['layers'][i]['config']['depth_multiplier']) +
                                   ' ' + str(l['config']['layers'][i]["config"]["use_bias"]) + '\n')
                            for o in range(W.shape[3]):
                                for k in range(W.shape[2]):
                                    for m in range(W.shape[0]):
                                        for n in range(W.shape[1]):
                                            fout.write(str(W[m, n, k, o]) + ',')
                                    fout.write('\n')
                                if l['config']['layers'][i]["config"]["use_bias"] == True:
                                    fout.write(str(b[o]) + '\n')
                        elif save_type == 'binary':
                            fout.write(struct.pack('i', W.shape[0]))
                            fout.write(struct.pack('i', W.shape[1]))
                            fout.write(struct.pack('i', W.shape[2]))
                            fout.write(struct.pack('i', W.shape[3]))
                            fout.write(struct.pack(
                                '>B', l['config']['layers'][i]['config']['kernel_size'][0]))
                            fout.write(struct.pack(
                                '>B', l['config']['layers'][i]['config']['kernel_size'][1]))
                            fout.write(struct.pack(
                                '>B', l['config']['layers'][i]['config']['strides'][0]))
                            fout.write(struct.pack(
                                '>B', l['config']['layers'][i]['config']['strides'][1]))
                            fout.write(struct.pack(
                                '>B', padding_dic[l['config']['layers'][i]['config']['padding']]))
                            fout.write(struct.pack(
                                '>B', l['config']['layers'][i]['config']['depth_multiplier']))
                            if l['config']['layers'][i]['config']['use_bias']:
                                fout.write(struct.pack('>B', 1))
                            else:
                                fout.write(struct.pack('>B', 0))
                            #print W.shape[0], W.shape[1], W.shape[2], W.shape[3]
                            for o in range(W.shape[3]):
                                for k in range(W.shape[2]):
                                    for m in range(W.shape[0]):
                                        for n in range(W.shape[1]):
                                            fout.write(struct.pack(
                                                'f', W[m, n, k, o]))
                                if l['config']['layers'][i]["config"]["use_bias"] == True:
                                    fout.write(struct.pack('f', b[o]))

        if save_type == 'txt':
            fout.write('End\n')
        elif save_type == 'binary':
            fout.write(struct.pack('>B', layer_dic["End"]))
    fout.close()


if __name__ == '__main__':
    model_type = 0  # config and weights in one single file
    """
    if model_type == 1:
        load_model_name = '../models_class/save_model.hdf5'
    model_name = '../models_class/save_model.json'
    weights_name = '../models_class/save_model.h5'
    save_name_txt = '../models_class/save_model.txt'
    save_name_binary = '../models_class/save_model.dat'
    """
    
    if model_type == 1:
        load_model_name = '../models_od/tiny_yolo_ocr_7.hdf5'
    model_name = '../models_od/tiny_yolo_ocr_7.json'
    weights_name = '../models_od/tiny_yolo_ocr_7.h5'
    save_name_txt = '../models_od/tiny_yolo_ocr_7.txt'
    save_name_binary = '../models_od/tiny_yolo_ocr_7.dat'    
    """
    
    if model_type == 1:
        load_model_name = '../models_od/mobilenet.hdf5'
    model_name = '../models_od/mobilenet.json'
    weights_name = '../models_od/mobilenet.h5'
    save_name_txt = '../models_od/mobilenet.txt'
    save_name_binary = '../models_od/mobilenet.dat'
    """
    if model_type == 1:
        load_save_model(load_model_name, model_name, weights_name)
    save_weights('txt', json_name=model_name, weights=weights_name, output=save_name_txt)
    save_weights('binary', json_name=model_name, weights=weights_name, output=save_name_binary)

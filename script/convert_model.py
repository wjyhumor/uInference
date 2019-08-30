# convert Keras model to binary file
from keras.models import model_from_json
from keras.models import Sequential
from keras.models import load_model
import json
import struct

layer_dic = {"Conv2D": 1, "BatchNormalization": 2, "Activation": 3, "MaxPooling2D": 4,
             "Flatten": 5, "Dense": 6, "InputLayer": 7, "Model": 8, "Reshape": 9,
             "Lambda": 10, "LeakyReLU": 11}
padding_dic = {"valid": 1, "same": 2}
activation_dic = {"relu": 1, "softmax": 2}

# load single-file model and save model as separate fils: json & weights
def load_save_model(load_model_name, save_model_name, save_weights_name):
    model = load_model(str(load_model_name))
    print(model.summary())
    model_json = model.to_json()
    with open(save_model_name, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(save_weights_name)

# save model weights as self-defined format txt file
def save_classification_txt(json_name="save_model.json", weights="save_model.h5", output="save_model.dat"):
    arch = open(json_name).read()
    model = model_from_json(arch)
    model.load_weights(weights)
    print(model.summary())
    arch = json.loads(arch)
    with open(output, 'w') as fout:
        fout.write('layers ' + str(len(model.layers)) + '\n')
        layers = []
        for ind, l in enumerate(arch["config"]["layers"]):
            print('layer ' + str(ind) + ' ' + l['class_name'])
            fout.write('layer ' + str(ind) + ' ' + l['class_name'] + '\n')
            layers += [l['class_name']]
            if l['class_name'] == 'Conv2D':
                W = model.layers[ind].get_weights()[0]
                b = model.layers[ind].get_weights()[1]
                fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) +
                           ' ' + str(W.shape[2]) + ' ' + str(W.shape[3]) +
                           ' ' + str(l['config']['padding']) + '\n')
                for l in range(W.shape[3]):
                    for k in range(W.shape[2]):
                        for i in range(W.shape[0]):
                            for j in range(W.shape[1]):
                                fout.write(str(W[i, j, k, l]) + ',')
                        fout.write('\n')
                    fout.write(str(b[l]) + '\n')
            elif l['class_name'] == 'BatchNormalization':
                gamma = model.layers[ind].get_weights()[0]
                bata = model.layers[ind].get_weights()[1]
                mean = model.layers[ind].get_weights()[2]
                variance = model.layers[ind].get_weights()[3]
                for i in range(0, len(gamma)):
                    fout.write(str(gamma[i]) + ',')
                    fout.write(str(bata[i]) + ',')
                    fout.write(str(mean[i]) + ',')
                    fout.write(str(variance[i]) + '\n')
            elif l['class_name'] == 'Activation':
                fout.write(l['config']['activation'] + '\n')
            elif l['class_name'] == 'MaxPooling2D':
                fout.write(str(l['config']['pool_size'][0]) +
                           ' ' + str(l['config']['pool_size'][1]) +
                           ' ' + str(l['config']['padding']) + '\n')
            elif l['class_name'] == 'Flatten':
                print(l['config']['name'])
            elif l['class_name'] == 'Dense':
                W = model.layers[ind].get_weights()[0]
                b = model.layers[ind].get_weights()[1]
                fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + '\n')
                for j in range(W.shape[1]):
                    for i in range(W.shape[0]):
                        fout.write(str(W[i, j]) + ',')
                    fout.write('\n')
                    fout.write(str(b[j]) + '\n')
    fout.close()

# save model weights as self-defined format binary file
def save_classification_binary(json_name="save_model.json", weights="save_model.h5", output="save_model.dat"):
    arch = open(json_name).read()
    model = model_from_json(arch)
    model.load_weights(weights)
    print(model.summary())
    arch = json.loads(arch)
    with open(output, 'wb') as fout:
        fout.write(struct.pack('>B', len(model.layers)))
        #fout.write('layers ' + str(len(model.layers)) + '\n')
        #layers = []
        for ind, l in enumerate(arch["config"]["layers"]):
            print('layer ' + str(ind) + ' ' + l['class_name'])
            fout.write(struct.pack('>B', layer_dic[l['class_name']]))
            #fout.write('layer ' + str(ind) + ' ' + l['class_name'] + '\n')
            #layers += [l['class_name']]
            if l['class_name'] == 'Conv2D':
                W = model.layers[ind].get_weights()[0]
                b = model.layers[ind].get_weights()[1]
                fout.write(struct.pack('>B', W.shape[0]))
                fout.write(struct.pack('>B', W.shape[1]))
                fout.write(struct.pack('>B', W.shape[2]))
                fout.write(struct.pack('>B', W.shape[3]))
                fout.write(struct.pack('>B', l['config']['strides'][0]))
                fout.write(struct.pack(
                    '>B', padding_dic[l['config']['padding']]))
                # fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) +
                #           ' ' + str(W.shape[2]) + ' ' + str(W.shape[3]) +
                #           ' ' + str(l['config']['padding']) + '\n')
                for l in range(W.shape[3]):
                    for k in range(W.shape[2]):
                        for i in range(W.shape[0]):
                            for j in range(W.shape[1]):
                                fout.write(struct.pack('f', W[i, j, k, l]))
                                #fout.write(str(W[i, j, k, l]) + ',')
                        # fout.write('\n')
                    fout.write(struct.pack('f', b[l]))
                    #fout.write(str(b[l]) + '\n')
            elif l['class_name'] == 'BatchNormalization':
                gamma = model.layers[ind].get_weights()[0]
                beta = model.layers[ind].get_weights()[1]
                mean = model.layers[ind].get_weights()[2]
                variance = model.layers[ind].get_weights()[3]
                for i in range(0, len(gamma)):
                    fout.write(struct.pack('f', gamma[i]))
                    fout.write(struct.pack('f', beta[i]))
                    fout.write(struct.pack('f', mean[i]))
                    fout.write(struct.pack('f', variance[i]))
                    #fout.write(str(gamma[i]) + ',')
                    #fout.write(str(bata[i]) + ',')
                    #fout.write(str(mean[i]) + ',')
                    #fout.write(str(variance[i]) + '\n')
            elif l['class_name'] == 'Activation':
                fout.write(struct.pack(
                    '>B', activation_dic[l['config']['activation']]))
                #fout.write(l['config']['activation'] + '\n')
            elif l['class_name'] == 'MaxPooling2D':
                fout.write(struct.pack('>B', l['config']['pool_size'][0]))
                fout.write(struct.pack('>B', l['config']['pool_size'][1]))
                fout.write(struct.pack(
                    '>B', padding_dic[l['config']['padding']]))
                # fout.write(str(l['config']['pool_size'][0]) +
                #           ' ' + str(l['config']['pool_size'][1]) +
                #           ' ' + str(l['config']['padding']) + '\n')
            elif l['class_name'] == 'Flatten':
                print(l['config']['name'])
            elif l['class_name'] == 'Dense':
                W = model.layers[ind].get_weights()[0]
                b = model.layers[ind].get_weights()[1]
                fout.write(struct.pack('>B', W.shape[0]))
                fout.write(struct.pack('>B', W.shape[1]))
                #fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + '\n')
                for j in range(W.shape[1]):
                    for i in range(W.shape[0]):
                        fout.write(struct.pack('f', W[i, j]))
                        #fout.write(str(W[i, j]) + ',')
                    # fout.write('\n')
                    fout.write(struct.pack('f', b[j]))
                    #fout.write(str(b[j]) + '\n')
    fout.close()

# 
def save_objectdetection_txt(json_name="save_model.json", weights="save_model.h5", output="save_model.dat"):
    arch = open(json_name).read()
    model = model_from_json(arch)
    model.load_weights(weights)
    print(model.summary())
    arch = json.loads(arch)
    with open(output, 'w') as fout:
        #fout.write('layers ' + str(len(model.layers)) + '\n')
        layers = []
        for ind, l in enumerate(arch["config"]["layers"]):
            print('layer ' + str(ind) + ' ' + l['class_name'])
            fout.write(l['class_name'] + '\n')
            layers += [l['class_name']]
            if l['class_name'] == 'Conv2D':
                W = model.layers[ind].get_weights()[0]
                if l['config']['use_bias'] == True:
                    b = model.layers[ind].get_weights()[1]
                fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) +
                           ' ' + str(W.shape[2]) + ' ' + str(W.shape[3]) +
                           ' ' + str(l['config']['strides'][0]) + 
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
            elif l['class_name'] == 'BatchNormalization':
                gamma = model.layers[ind].get_weights()[0]
                bata = model.layers[ind].get_weights()[1]
                mean = model.layers[ind].get_weights()[2]
                variance = model.layers[ind].get_weights()[3]
                for i in range(0, len(gamma)):
                    fout.write(str(gamma[i]) + ',')
                    fout.write(str(bata[i]) + ',')
                    fout.write(str(mean[i]) + ',')
                    fout.write(str(variance[i]) + '\n')
            elif l['class_name'] == 'Activation':
                fout.write(l['config']['activation'] + '\n')
            elif l['class_name'] == 'MaxPooling2D':
                fout.write(str(l['config']['pool_size'][0]) +
                           ' ' + str(l['config']['pool_size'][1]) +
                           ' ' + str(l['config']['padding']) + '\n')
            elif l['class_name'] == 'Flatten':
                print(l['config']['name'])
            elif l['class_name'] == 'Dense':
                W = model.layers[ind].get_weights()[0]
                b = model.layers[ind].get_weights()[1]
                fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + '\n')
                for j in range(W.shape[1]):
                    for i in range(W.shape[0]):
                        fout.write(str(W[i, j]) + ',')
                    fout.write('\n')
                    fout.write(str(b[j]) + '\n')
            elif l['class_name'] == 'Model':
                #print(len(l['config']['layers']))
                sublayer_ind = 0
                for i in range(len(l['config']['layers'])):
                    name = l['config']['layers'][i]["class_name"]
                    print(name)
                    fout.write(name + '\n')
                    if name == 'Conv2D':
                        W = model.layers[ind].get_weights()[sublayer_ind]
                        sublayer_ind += 1
                        if l['config']['layers'][i]["config"]["use_bias"] == True:
                            b = model.layers[ind].get_weights()[sublayer_ind]
                            sublayer_ind += 1
                        fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) +
                                ' ' + str(W.shape[2]) + ' ' + str(W.shape[3]) +
                                ' ' + str(l['config']['layers'][i]['config']['strides'][0]) +
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
                    elif name == 'BatchNormalization':
                        gamma = model.layers[ind].get_weights()[sublayer_ind]
                        sublayer_ind += 1
                        beta = model.layers[ind].get_weights()[sublayer_ind]
                        sublayer_ind += 1
                        mean = model.layers[ind].get_weights()[sublayer_ind]
                        sublayer_ind += 1
                        variance = model.layers[ind].get_weights()[sublayer_ind]
                        sublayer_ind += 1
                        for k in range(0, len(gamma)):
                            fout.write(str(gamma[k]) + ',')
                            fout.write(str(beta[k]) + ',')
                            fout.write(str(mean[k]) + ',')
                            fout.write(str(variance[k]) + '\n')
                    elif name == 'LeakyReLU':
                        fout.write(str(l['config']['layers'][i]['config']['alpha']) + '\n')
                    elif name == 'MaxPooling2D':
                        fout.write(str(l['config']['layers'][i]['config']['pool_size'][0]) +
                                ' ' + str(l['config']['layers'][i]['config']['pool_size'][1]) +
                                ' ' + str(l['config']['layers'][i]['config']['padding']) + '\n')     
                      
    fout.close()


# save model weights as self-defined format binary file
def save_objectdetection_binary(json_name="save_model.json", weights="save_model.h5", output="save_model.dat"):
    arch = open(json_name).read()
    model = model_from_json(arch)
    model.load_weights(weights)
    print(model.summary())
    arch = json.loads(arch)
    with open(output, 'wb') as fout:
        #fout.write(struct.pack('>B', len(model.layers)))
        #fout.write('layers ' + str(len(model.layers)) + '\n')
        #layers = []
        for ind, l in enumerate(arch["config"]["layers"]):
            print('layer ' + str(ind) + ' ' + l['class_name'])
            fout.write(struct.pack('>B', layer_dic[l['class_name']]))
            #fout.write('layer ' + str(ind) + ' ' + l['class_name'] + '\n')
            #layers += [l['class_name']]
            if l['class_name'] == 'Conv2D':
                W = model.layers[ind].get_weights()[0]
                if l['config']['use_bias'] == True:
                    b = model.layers[ind].get_weights()[1]
                fout.write(struct.pack('>B', W.shape[0]))
                fout.write(struct.pack('>B', W.shape[1]))
                fout.write(struct.pack('>B', W.shape[2]))
                fout.write(struct.pack('>B', W.shape[3]))
                fout.write(struct.pack('>B', l['config']['strides'][0]))
                fout.write(struct.pack(
                    '>B', padding_dic[l['config']['padding']]))
                if l['config']['use_bias']:
                    fout.write(struct.pack('>B', 1))
                else:
                    fout.write(struct.pack('>B', 0))
                
                # fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) +
                #           ' ' + str(W.shape[2]) + ' ' + str(W.shape[3]) +
                #           ' ' + str(l['config']['padding']) + '\n')
                for o in range(W.shape[3]):
                    for k in range(W.shape[2]):
                        for i in range(W.shape[0]):
                            for j in range(W.shape[1]):
                                fout.write(struct.pack('f', W[i, j, k, o]))
                                #fout.write(str(W[i, j, k, l]) + ',')
                        # fout.write('\n')                    
                    if l['config']['use_bias'] == True:
                        fout.write(struct.pack('f', b[o]))
                    #fout.write(str(b[l]) + '\n')
            elif l['class_name'] == 'BatchNormalization':
                gamma = model.layers[ind].get_weights()[0]
                beta = model.layers[ind].get_weights()[1]
                mean = model.layers[ind].get_weights()[2]
                variance = model.layers[ind].get_weights()[3]
                for i in range(0, len(gamma)):
                    fout.write(struct.pack('f', gamma[i]))
                    fout.write(struct.pack('f', beta[i]))
                    fout.write(struct.pack('f', mean[i]))
                    fout.write(struct.pack('f', variance[i]))
                    #fout.write(str(gamma[i]) + ',')
                    #fout.write(str(bata[i]) + ',')
                    #fout.write(str(mean[i]) + ',')
                    #fout.write(str(variance[i]) + '\n')
            elif l['class_name'] == 'Activation':
                fout.write(struct.pack(
                    '>B', activation_dic[l['config']['activation']]))
                #fout.write(l['config']['activation'] + '\n')
            elif l['class_name'] == 'MaxPooling2D':
                fout.write(struct.pack('>B', l['config']['pool_size'][0]))
                fout.write(struct.pack('>B', l['config']['pool_size'][1]))
                fout.write(struct.pack(
                    '>B', padding_dic[l['config']['padding']]))
                # fout.write(str(l['config']['pool_size'][0]) +
                #           ' ' + str(l['config']['pool_size'][1]) +
                #           ' ' + str(l['config']['padding']) + '\n')
            elif l['class_name'] == 'Flatten':
                print(l['config']['name'])
            elif l['class_name'] == 'Dense':
                W = model.layers[ind].get_weights()[0]
                b = model.layers[ind].get_weights()[1]
                fout.write(struct.pack('>B', W.shape[0]))
                fout.write(struct.pack('>B', W.shape[1]))
                #fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + '\n')
                for j in range(W.shape[1]):
                    for i in range(W.shape[0]):
                        fout.write(struct.pack('f', W[i, j]))
                        #fout.write(str(W[i, j]) + ',')
                    # fout.write('\n')
                    fout.write(struct.pack('f', b[j]))
                    #fout.write(str(b[j]) + '\n')
            elif l['class_name'] == 'Model':
                #print(len(l['config']['layers']))
                sublayer_ind = 0
                for i in range(len(l['config']['layers'])):
                    name = l['config']['layers'][i]["class_name"]
                    print(name)
                    fout.write(struct.pack('>B', layer_dic[name]))
                    #fout.write(name + '\n')
                    if name == 'Conv2D':
                        W = model.layers[ind].get_weights()[sublayer_ind]
                        sublayer_ind += 1
                        if l['config']['layers'][i]["config"]["use_bias"] == True:
                            b = model.layers[ind].get_weights()[sublayer_ind]
                            sublayer_ind += 1
                        fout.write(struct.pack('>B', W.shape[0]))
                        fout.write(struct.pack('>B', W.shape[1]))
                        fout.write(struct.pack('>B', W.shape[2]))
                        fout.write(struct.pack('>B', W.shape[3]))
                        fout.write(struct.pack('>B', l['config']['layers'][i]['config']['strides'][0]))
                        fout.write(struct.pack(
                            '>B', padding_dic[l['config']['layers'][i]['config']['padding']]))
                        if l['config']['layers'][i]['config']['use_bias']:
                            fout.write(struct.pack('>B', 1))
                        else:
                            fout.write(struct.pack('>B', 0))
                        for o in range(W.shape[3]):
                            for k in range(W.shape[2]):
                                for m in range(W.shape[0]):
                                    for n in range(W.shape[1]):
                                        fout.write(struct.pack('f', W[m, n, k, o]))
                            if l['config']['layers'][i]["config"]["use_bias"] == True:
                                fout.write(struct.pack('f', b[o]))
                    elif name == 'BatchNormalization':
                        gamma = model.layers[ind].get_weights()[sublayer_ind]
                        sublayer_ind += 1
                        beta = model.layers[ind].get_weights()[sublayer_ind]
                        sublayer_ind += 1
                        mean = model.layers[ind].get_weights()[sublayer_ind]
                        sublayer_ind += 1
                        variance = model.layers[ind].get_weights()[sublayer_ind]
                        sublayer_ind += 1
                        for k in range(0, len(gamma)):
                            fout.write(struct.pack('f', gamma[k]))
                            fout.write(struct.pack('f', beta[k]))
                            fout.write(struct.pack('f', mean[k]))
                            fout.write(struct.pack('f', variance[k]))
                    elif name == 'LeakyReLU':
                        fout.write(struct.pack('f', l['config']['layers'][i]['config']['alpha']))
                    elif name == 'MaxPooling2D':   
                        fout.write(struct.pack('>B', l['config']['layers'][i]['config']['pool_size'][0]))
                        fout.write(struct.pack('>B', l['config']['layers'][i]['config']['pool_size'][1]))
                        fout.write(struct.pack(
                            '>B', padding_dic[l['config']['layers'][i]['config']['padding']])) 
  
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

    if model_type == 1:
        load_save_model(load_model_name, model_name, weights_name)
    save_classification_txt(
        json_name=model_name, weights=weights_name, output=save_name_txt)
    save_classification_binary(
        json_name=model_name, weights=weights_name, output=save_name_binary)
    """
    if model_type == 1:
        load_model_name = '../models_od/tiny_yolo_ocr_6.hdf5'
    model_name = '../models_od/tiny_yolo_ocr_6.json'
    weights_name = '../models_od/tiny_yolo_ocr_6.h5'
    save_name_txt = '../models_od/tiny_yolo_ocr_6.txt'
    save_name_binary = '../models_od/tiny_yolo_ocr_6.dat'

    if model_type == 1:
        load_save_model(load_model_name, model_name, weights_name)
    save_objectdetection_txt(
        json_name=model_name, weights=weights_name, output=save_name_txt)
    save_objectdetection_binary(
        json_name=model_name, weights=weights_name, output=save_name_binary)
    
    

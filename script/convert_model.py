
from keras.models import model_from_json
from keras.models import Sequential
from keras.models import load_model
import json

def load(json_name="save_model.json", weights="save_model.h5", output="save_model.dat"):
    arch = open(json_name).read()
    model = model_from_json(arch)
    model.load_weights(weights)
    print(model.summary())
    arch = json.loads(arch)

    with open(output, 'w') as fout:
        fout.write('layers ' + str(len(model.layers)) + '\n')
        layers = []
        for ind, l in enumerate(arch["config"]["layers"]):
            fout.write('layer ' + str(ind) + ' ' + l['class_name'] + '\n')
            layers += [l['class_name']]
            if l['class_name'] == 'Conv2D':
                W = model.layers[ind].get_weights()[0]
                fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + ' ' + str(W.shape[2]) + ' ' + str(W.shape[3]) + '\n')
                for i in range(W.shape[0]):
                    for j in range(W.shape[1]):
                        for k in range(W.shape[2]):
                            fout.write(str(W[i,j,k]) + '\n')
                fout.write(str(model.layers[ind].get_weights()[1]) + '\n')
            if l['class_name'] == 'Activation':
                fout.write(l['config']['activation'] + '\n')
            if l['class_name'] == 'MaxPooling2D':
                fout.write(str(l['config']['pool_size'][0]) + ' ' + str(l['config']['pool_size'][1]) + '\n')
            if l['class_name'] == 'Flatten':
                print(l['config']['name'])
            if l['class_name'] == 'BatchNormalization':
                W = model.layers[ind].get_weights()
                for w in W:
                    fout.write(str(w) + '\n')
            if l['class_name'] == 'Dense':
                #fout.write(str(l['config']['output_dim']) + '\n')
                W = model.layers[ind].get_weights()[0]
                fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + '\n')
                for w in W:
                    fout.write(str(w) + '\n')
                fout.write(str(model.layers[ind].get_weights()[1]) + '\n')


def load_save_model(load_model_name, save_model_name, save_weights_name):
    model = load_model(str(load_model_name))
    print(model.summary())
    model_json = model.to_json()
    with open(save_model_name, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(save_weights_name)

if __name__ == '__main__':
    model_type = 1 # config and weights in one file
    if model_type == 1:
        load_model_name = '../models/weights-model1.hdf5'

    save_model_name = '../models/save_model.json'
    save_weights_name = '../models/save_model.h5'
    output = '../models/save_model.dat'

    if model_type == 1:
        load_save_model(load_model_name, save_model_name, save_weights_name)
    
    load(json_name=save_model_name, weights=save_weights_name, output=output)
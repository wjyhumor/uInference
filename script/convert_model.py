# convert Keras model to dat file
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


def load_save_model(load_model_name, save_model_name, save_weights_name):
    model = load_model(str(load_model_name))
    print(model.summary())
    model_json = model.to_json()
    with open(save_model_name, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(save_weights_name)


if __name__ == '__main__':
    model_type = 0  # config and weights in one file
    if model_type == 1:
        load_model_name = '../models/weights-model1.hdf5'

    save_model_name = '../models/save_model.json'
    save_weights_name = '../models/save_model.h5'
    output = '../models/save_model.dat'

    if model_type == 1:
        load_save_model(load_model_name, save_model_name, save_weights_name)

    load(json_name=save_model_name, weights=save_weights_name, output=output)

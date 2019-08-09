# keras model to C++

This code is to port Keras neural network model to C++. Neural networks architecture is saved in a JSON file and weights are stored in HDF5 format. The saved model is then loaded and dumped to .dat file, which will be used in cpp file. As of now the it supports Dense and Activation layers only. Also you have to add activation as a new layer instead of passing it as a parameter to Dense layer. Major activation functions like `relu, softmax, sigmoid, linear, softplus, softsign, etc` has been implemented. You can add implementation to any other activation function or any other layer as you wish in future.
Also the code works for binary classification only as it was my only requirement at that time. But it can be easily extended for multiclass classification.  

It is working with the Tensorflow backend.

# Usage

 1. Save your network weights and architecture using `save_model.py` script.
 2. Dump network structure to plain text file with `dump_to_cpp.py` script.
 3. Use network with code from `predict.h` and `predict.cc` files - see `test_run.sh`.

# Example

 1. Run `save_model.py` script. It will produce files with architecture `arch.json` and weights in HDF5 format `weights.h5`.
 2. Dump network to dat file `python2 convert_to_cpp.py -a save_model.json -w save_weights.h5 -o save.dat`.
 3. Run `test_keras.py -a save_model.json -w save_weights.h5 -i save.dat`. It will output the predict and actual class based on data in `input.dat` file.
 4. Compile example `g++ test_main.cpp predict.cpp` - see code in `test_main.cpp`.
 5. Run binary `./a.out dumped_nn.dat input.dat` - you should get the same output as in step one from Keras.

# Testing

If you want to test dumping for your network, please use `test_run.sh` script. Please provide there your network architecture and weights. The script do following job:

 1. Dump network into text file.
 2. Generate random sample and save in inout.dat. First line contains number of features in input, next line contains features and last line contains actual class.
 3. Compute predictions from keras and cpp on generated sample.
 4. Compare predictions.

# DNN
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 16, 16, 8)         208       
_________________________________________________________________
batch_normalization_1 (Batch (None, 16, 16, 8)         32        
_________________________________________________________________
activation_1 (Activation)    (None, 16, 16, 8)         0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 8, 8)           0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 8, 8, 16)          3216      
_________________________________________________________________
batch_normalization_2 (Batch (None, 8, 8, 16)          64        
_________________________________________________________________
activation_2 (Activation)    (None, 8, 8, 16)          0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 2, 2, 16)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 64)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                650       
_________________________________________________________________
activation_3 (Activation)    (None, 10)                0         
=================================================================
Total params: 4,170
Trainable params: 4,122
Non-trainable params: 48


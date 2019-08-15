# What is uInference?
uInference is a project which could run DNN on micro-chip like STM32 etc.

# Usage
1. Run `python2 convert_model.py` to convert the model to the .dat file.
2. make to create, and run the bin.
3. Run `predict.py` to run the example.jpg, compare the results.


# DNN model example
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

##  
Memory used:15.59KB
weigths: 16.7KB

## Check leakage:
```
valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all -v ./uInference.bin
```
Check memory:
```
valgrind --tool=massif ./uInference.bin
ms_print massif.out.
```

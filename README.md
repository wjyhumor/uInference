# What is uInference?
uInference is a inference framework which could run Classification and Object Detection on a quite small micro-chip like STM32 etc.

# Usage
1. In folder `script/`, run `python img_save2binary.py` to convert the image to the .img file.
2. In folder `script/`, run `python convert_model.py` to convert the model to the .dat file.
3. In root folder, make and run the uInference.bin.
4. In folder `train_class/`, run `predict_class.py` to run the example.jpg, compare the results, you will see that the outputs are exactly the same.

# Classification
## Training (in `tran_class/`)
Run `python train.py`.

## DNN model example for classification
Layer (type)                 |Output Shape      |Param #   
-----------------------------|------------------|---------------
conv2d_1 (Conv2D)            |(None, 16, 16, 8) |208       
batch_normalization_1        |(None, 16, 16, 8) |32        
activation_1 (Activation)    |(None, 16, 16, 8) |0         
max_pooling2d_1 (MaxPooling2 |(None, 8, 8, 8)   |0         
conv2d_2 (Conv2D)            |(None, 8, 8, 16)  |3216      
batch_normalization_2 (Batch |(None, 8, 8, 16)  |64        
activation_2 (Activation)    |(None, 8, 8, 16)  |0         
max_pooling2d_2 (MaxPooling2 |(None, 2, 2, 16)  |0         
flatten_1 (Flatten)          |(None, 64)        |0         
dense_1 (Dense)              |(None, 10)        |650       
activation_3 (Activation)    |(None, 10)        |0         

Total params: 4,170  
Trainable params: 4,122  
Non-trainable params: 48  

* RAM:15.59KB   
* ROM: 16.7KB
* Speed: 5.577000 ms @Intel(R) Core(TM) i5-7500 CPU @ 3.40GHz

# Object Detection
## Training (in `tran_od/`)
1. Generate anchors for your dataset (optional)  
`python gen_anchors.py -c config.json`  
Copy the generated anchors printed on the terminal to the `anchors` setting in `config.json`.

2. Start the training process  
`python train.py -c config.json`

3. Perform detection using trained weights on an image by running
`python predict.py -c config.json -i /path/to/image/or/video`  
example:  
`python predict.py -c config.json -i ../ex_od.jpg`  

## Models 
tiny_yolo_ocr_layer3ch.h5: TinyYoloFeature, channel=3; Model size=189836484  
tiny_yolo_ocr_layer1ch.h5: TinyYoloFeature, channel=1; Model size=189833856  
tiny_yolo_ocr_1.h5: TinyYoloFeature_1, input_size=320; Model size=12138400  
tiny_yolo_ocr_2.h5: TinyYoloFeature_2, input_size=320; Model size=3163852  
tiny_yolo_ocr_3.h5: TinyYoloFeature_3, input_size=320; Model size=1383484  
tiny_yolo_ocr_4.h5: TinyYoloFeature_4, input_size=320; Model size=891676  
tiny_yolo_ocr_5.h5: TinyYoloFeature_4, input_width=320, input_height=105; Model size=891676  
tiny_yolo_ocr_6.h5: TinyYoloFeature_5, input_width=320, input_height=105; Model size=635004  

## DNN model example for object detection
Layer (type)                 |Output Shape              |Param #   
-----------------------------|--------------------------|----------
input_3 (InputLayer)         |(None, 105, 320, 1)       |0         
conv_1 (Conv2D)              |(None, 105, 320, 4)       |36        
norm_1 (BatchNormalization)  |(None, 105, 320, 4)       |16        
leaky_re_lu_1 (LeakyReLU)    |(None, 105, 320, 4)       |0         
max_pooling2d_1 (MaxPooling2 |(None, 52, 160, 4)        |0         
conv_2 (Conv2D)              |(None, 52, 160, 4)        |144       
norm_2 (BatchNormalization)  |(None, 52, 160, 4)        |16        
leaky_re_lu_2 (LeakyReLU)    |(None, 52, 160, 4)        |0         
max_pooling2d_2 (MaxPooling2 |(None, 26, 80, 4)         |0         
conv_3 (Conv2D)              |(None, 26, 80, 8)         |288       
norm_3 (BatchNormalization)  |(None, 26, 80, 8)         |32        
leaky_re_lu_3 (LeakyReLU)    |(None, 26, 80, 8)         |0         
max_pooling2d_3 (MaxPooling2 |(None, 13, 40, 8)         |0        
conv_4 (Conv2D)              |(None, 13, 40, 16)        |1152      
norm_4 (BatchNormalization)  |(None, 13, 40, 16)        |64        
leaky_re_lu_4 (LeakyReLU)    |(None, 13, 40, 16)        |0         
max_pooling2d_4 (MaxPooling2 |(None, 6, 20, 16)         |0         
conv_5 (Conv2D)              |(None, 6, 20, 32)         |4608      
norm_5 (BatchNormalization)  |(None, 6, 20, 32)         |128       
leaky_re_lu_5 (LeakyReLU)    |(None, 6, 20, 32)         |0         
max_pooling2d_5 (MaxPooling2 |(None, 3, 10, 32)         |0         
conv_6 (Conv2D)              |(None, 3, 10, 64)         |18432     
norm_6 (BatchNormalization)  |(None, 3, 10, 64)         |256       
leaky_re_lu_6 (LeakyReLU)    |(None, 3, 10, 64)         |0         
max_pooling2d_6 (MaxPooling2 |(None, 3, 10, 64)         |0         
conv_10 (Conv2D)             |(None, 3, 10, 64)         |16384     
norm_10 (BatchNormalization) |(None, 3, 10, 64)         |256       
leaky_re_lu_7 (LeakyReLU)    |(None, 3, 10, 64)         |0         

Total params: 41,812  
Trainable params: 41,428  
Non-trainable params: 384  

Layer (type)                 |Output Shape         |Param #     |Connected to
-----------------------------|---------------------|------------|-------------
input_1 (InputLayer)         |(None, 105, 320, 1)  |0           |        
model_1 (Model)              |(None, 3, 10, 64)    |41812       |input_1[0][0] 
DetectionLayer (Conv2D)      |(None, 3, 10, 75)    |4875        |model_1[1][0]    
reshape_1 (Reshape)          |(None, 3, 10, 5, 15) |0           |DetectionLayer[0][0]       
input_2 (InputLayer)         |(None, 1, 1, 1, 10,  |0                                  
lambda_1 (Lambda)            |(None, 3, 10, 5, 15) |0           |reshape_1[0][0],  input_2[0][0]   

Total params: 46,687  
Trainable params: 46,303  
Non-trainable params: 384  

RAM: 661.9 KB
ROM: 186.9 KB
Speed: 20 ms

# Check leakage:
```
valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all -v ./uInference.bin
```
Check memory:
```
valgrind --tool=massif ./uInference.bin
ms_print massif.out.
```

# Basic terms:
* Epochs: One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.

* Batch Size: Total number of training examples present in a single batch. 
Batch_size will not affect accuracy too much, but higher batch_size will need more memory when training.

* Iterations(Steps): the number of batches needed to complete one epoch.

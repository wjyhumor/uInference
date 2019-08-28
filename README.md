# What is uInference?
uInference is a inference framework which could run Classification and Object Detection on a quite small micro-chip like STM32 etc.

# Usage
1. In folder `script/`, run `python convert_model_binary.py` to convert the model to the .dat file.
2. In folder `script/`, run `python img_save2binary.py` to convert the image to the .img file.
3. In root folder, make and run the uInference.bin.
4. In folder `script/`, run `predict.py` to run the example.jpg, compare the results.

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

Memory used:15.59KB   
weigths (save_model_binary): 16.7KB


# Object Detection
## Training (in `tran_od/`)
1. Generate anchors for your dataset (optional)  
`python gen_anchors.py -c config.json`  
Copy the generated anchors printed on the terminal to the `anchors` setting in `config.json`.

2. Start the training process  
`python train.py -c config.json`

3. Perform detection using trained weights on an image by running
`python predict.py -c config.json -w /path/to/weights.h5 -i /path/to/image/or/video`  
`python predict.py -c config.json -w ../models_od/tiny_yolo_ocr_5.h5 -i ../ex_od.jpg`

## Models
tiny_yolo_ocr_layer3ch.h5: TinyYoloFeature, channel=3; Model size=189836484  
tiny_yolo_ocr_layer1ch.h5: TinyYoloFeature, channel=1; Model size=189833856  
tiny_yolo_ocr_1.h5: TinyYoloFeature_1, input_size=320; Model size=12138400  
tiny_yolo_ocr_2.h5: TinyYoloFeature_2, input_size=320; Model size=3163852  
tiny_yolo_ocr_3.h5: TinyYoloFeature_3, input_size=320; Model size=1383484  
tiny_yolo_ocr_4.h5: TinyYoloFeature_4, input_size=320; Model size=891676  
tiny_yolo_ocr_5.h5: TinyYoloFeature_4, input_width=320, input_height=105; Model size=891676  
tiny_yolo_ocr_6.h5: TinyYoloFeature_5, input_width=320, input_height=105; Model size=635004  


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

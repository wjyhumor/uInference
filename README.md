# What is uInference?
uInference is a inference framework which could run Classification and Object Detection on a extremely small micro-chip like STM32 etc.

# Usage
1. In folder `script/`, run `python img_save2binary.py` to convert the image to the .img file.
2. In folder `script/`, run `python convert_model.py` to convert the model to the .dat file.
3. In root folder, make and run the uInference.bin.
4. In folder `train_class/`, run `predict_class.py` to run the example.jpg, compare the results, you will see that the outputs are exactly the same.

# Classification
## Training (in `train_class/`)
Run `python train.py`.

# Object Detection
## Training (in `train_od/`)
1. Generate anchors for your dataset (optional)  
`python gen_anchors.py -c config.json`  
Copy the generated anchors printed on the terminal to the `anchors` setting in `config.json`.

2. Start the training process  
`python train.py -c config.json`

3. Perform detection using trained weights on an image by running
`python predict.py -c config.json -i /path_to_image`  
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
tiny_yolo_ocr_7.h5: TinyYoloFeature_5, input_width=320, input_height=105; Model size=635004, the same as *_6, but with pure structure.  

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
Speed: 19.558001 ms @Intel(R) Core(TM) i5-7500 CPU @ 3.40GHz  

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

* Keras data format: (samples, rows, cols, channels).  

# ROM limit:
<=150 KB

# Tests  
* model_1(Total params: 4,170)  
16 x 16:(2 old electric meter training data)   
Train accuracy:0.9722907010613903  
water_mechanical_190625:0.7128016085790885  
water_electric_190911/pre:**0.5761787245135476**  
water_electric_190911/last:0.38145859088974327  
electric_digital_190329:0.6264975334795282  
28 x 40:0.5432305630026809   

* model_2(Total params: 4,994)  
16 x 16:(2 old electric meter training data)   
Train accuracy:0.9818558327581106  
water_mechanical_190625:0.7310991957104558  
water_electric_190911/pre:**0.6880114300481095**  
water_electric_190911/last:0.4084054388207175  
28 x 40:0.6077077747989276  

* model_3(Total params: 9,762)   
1. 16 x 16: (2 old electric meter training data)  
Train accuracy:0.9864904841731584  
water_mechanical_190625:0.7914879356568365  
water_electric_190911/pre:**0.7585400701312373**  
water_electric_190911/last:0.4608158220245753  
2. **16 x 16: (2 old electric + 2 water meter training data, 0.8 data)**  
Train accuracy:0.9811729649517024  
water_electric_190911/pre:**0.7622418495985979**  
water_electric_190911/last:0.5636588381601058  
electric_digital_190329:0.6608527132202993  
3. 16 x 16: (2 new training data)  
Train accuracy:0.9421445101721792  
water_electric_190911/pre:0.9764255097987266  
water_electric_190911/last:0.8464771323357289  
electric_mechanical_190625:**0.8928113598382023**  
water_mechanical_190625:**0.7993297587131367**  
electric_digital_190329:**0.952959830824803**  
4. 16 x 16: (2 old electric + 2 water training data/pre)  
Train accuracy:0.996622192197264  
water_electric_190911/pre:0.7021691128640597  
water_electric_190911/last:0.35500618047339955   
5. 16 x 16: (2 old electric + 2 water + 2 new training data)  
Train accuracy:0.9667317246273953  
water_electric_190911/pre:**0.9746720353215217**  
water_electric_190911/last:**0.8370828183531319**  
electric_digital_190329:0.9044221283013423  
6. 16 x 16: (2 old electric + 2 water + 1 new training data/pre)  
Train accuracy:0.9898900122208644  
water_electric_190911/pre:**0.9777893232809969**  
water_electric_190911/last:0.45241038319280624  
7. 16 x 16: (2 old electric + 2 water + 1 new training data/last)  
Train accuracy:0.8947600172459423  
water_electric_190911/pre:0.7937394466891291  
water_electric_190911/last:**0.8707045735770604**  
8. 16 x 16: (2 old electric + 2 water + 1 digital meter training data)  
Train accuracy:0.9831158069194117  
water_electric_190911/pre:**0.8068580335108456**  
water_electric_190911/last:**0.5530284301975307**  
electric_digital_190402:0.9994416527079844  
9. **16 x 16: (2 old electric + 2 water meter training data, 0.4 data)**  
Train accuracy:0.9804957334416904  
water_electric_190911/pre:**0.7704896739836342**  
water_electric_190911/last:**0.5374536465213384**  
electric_digital_190329:**0.23185341792093186**  
10. **16 x 16: (2 old electric + 2 water meter training data, 0.99 data)**  
Train accuracy:0.9837837837837838  
water_electric_190911/pre:**0.8058189375166119**
water_electric_190911/last:**0.5601977751045792**
electric_digital_190402:**0.5676996091568955**
11. 28 x 40:  
Train accuracy:0.9467696059158658  
water_electric_190911/pre:0.5787115209844921  
water_electric_190911/last:0.3933250927107296  

* model_4(Total params: 10,994)  
1. 16 x 16: (2 old electric meter training data)  
Train accuracy:0.9820085929748708  
water_mechanical_190625:0.7976541554959785  
water_electric_190911/pre:**0.7362644499208203**  
water_electric_190911/last:0.4595797280814356  
2. 16 x 16: (2 old electric + 2 water meter training data, 0.8 data)  
Train accuracy:0.9827983204659353  
water_electric_190911/pre:**0.8188076373555007**  
water_electric_190911/last:0.6135970334187426  
3. 2.+dropout
Train accuracy:0.9818501963971286  
water_electric_190911/pre:**0.7930900116898298**  
water_electric_190911/last:0.6274412855598039  
4. 16 x 16: (2 new training data)  
Train accuracy:0.9521727950629982  
water_electric_190911/pre:0.9764255098064684  
water_electric_190911/last:0.8605686032580504  
electric_mechanical_190625:**0.9161818361225924**  
water_mechanical_190625:**0.8575067024128686**  
electric_digital_190329:**0.8430232558139535**  

* model_5(Total params: 14,530)   
1. 16 x 16: (2 old electric meter training data)    
Train accuracy:0.9787862514066645  
water_mechanical_190625:0.7731903485254692  
water_electric_190911/pre:**0.7290557215145338**  
water_electric_190911/last:0.456365883814537  
2. 16 x 16: (2 old electric + 2 water meter training data, 0.8 data)  
Train accuracy:0.9806311797372342  
water_electric_190911/pre:**0.7632809455850895**  
water_electric_190911/last:0.5977750309244516  

* model_6(Total params: 19,778)  
16 x 16:(2 old electric meter training data)   
Train accuracy:0.9828141783669223  
water_mechanical_190625:0.7833780160857908  
water_electric_190911/pre:**0.6948304974672035**  
water_electric_190911/last:0.4526576020072211  
16 x 16:(2 old electric + 2 water meter training data, 0.8 data)  
Train accuracy:0.9834755519436543  
water_electric_190911/pre:**0.7222366541186653**  
water_electric_190911/last:0.5260815822223504  

* model_7
16 x 16: (2 old electric + 2 water meter training data, 0.8 data)  
Train accuracy:0.9806311797372342  
water_electric_190911/pre:**0.791076763223744**  
water_electric_190911/last:0.5728059332877655  

* model_8
16 x 16: (2 old electric + 2 water meter training data, 0.8 data)  
Train accuracy:0.9823919815793039  
water_electric_190911/pre:**0.8203662813274964**  
water_electric_190911/last:0.5737948084938511

* model_9
16 x 16: (2 old electric + 2 water meter training data, 0.8 data)  
Train accuracy:0.9833401056481105  
water_electric_190911/pre:**0.7608130926171717**  
water_electric_190911/last:0.620024721923069

* model_server(Total params: 330,954)  
16 x 16:(2 old electric meter training data)     
Train accuracy:0.9891529434966966  
water_mechanical_190625:0.7783512064343163  
water_electric_190911/pre:**0.6745681257383562**  
water_electric_190911/last:0.5186650185487769  
28 x 40:0.5904155495978552  


* Summary of test:
1. Deeper is better, but not always better with certain amount of data (model_3 is the best now, deeper model 4, 5, 6 is worse).  
2. Should not test on the same dataset of OCR! The similarity of one dataset is too hight that it could not give the correct result of the performance of the model. Sould use TOTALLY different dataset for test!
3. More data is better!
4. Separate the previous digits with last digit will improve the model, but not too much(0.003 for pre, 0.04 for last).
5. Digital meters are mucher easier to classify than mechanical meters, don't mix digital and mechanical data to train.
6. 16x16 -> 28x40 will greatly decrease the accuracy, reason unknown.

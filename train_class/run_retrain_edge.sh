# retrain
python3 retrain.py \
 -type 1 \
 -pretrain 0 \
 -batch_size 16 \
 -epochs 50 \
 -reload_train 1 \
 -reload_test 1 \
 -origin_images ./tmp/images_base_all.list \
 -new_images   /home/neusoft/amy/uInference/data/beilu_0819/ \
 -save_model   ./tmp/weights_retrain_edge.hdf5 

python3 convert_model.py \
 -type 1 \
 -model_path ./tmp/weights_retrain_edge.hdf5 \
 -config_name ./tmp/weights_retrain_edge.json \
 -weights_name ./tmp/weights_retrain_edge.h5 \
 -save_name_txt ./tmp/weights_retrain_edge.txt \
 -save_name_binary ./tmp/weights_retrain_edge.dat


# CubeAI transform to ./tmp/cubeai/network_data.c 
#export X_CUBE_AI_DIR=/home/neusoft/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/4.0.0
#export PATH=$X_CUBE_AI_DIR/Utilities/linux:$PATH
#stm32ai analyze -m ./tmp/weights_retrain_edge.hdf5 --type keras
#stm32ai validate -m ./tmp/weights_retrain_edge.hdf5 --type keras
#stm32ai generate -m ./tmp/weights_retrain_edge.hdf5 --type keras \
#                 -o ./tmp/cubeai

# transform to .bin
#python3 model2bin_server.py \
# -input ./tmp/cubeai/network_data.c \
# -output ./tmp/weights_retrain_edge.bin

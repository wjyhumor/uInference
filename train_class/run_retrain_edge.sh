# retrain
python3 retrain.py \
 -type 1 \
 -pretrain 0 \
 -original_images_list /home/neusoft/amy/uInference/data/water_elec_0516_0625.digits.all \
 -new_images   /home/neusoft/amy/uInference/data/beilu_0819/ \
 -save_model   ./tmp/retrain_weights_edge.hdf5 

# CubeAI transform to ./tmp/cubeai/network_data.c 
export X_CUBE_AI_DIR=/home/neusoft/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/4.0.0
export PATH=$X_CUBE_AI_DIR/Utilities/linux:$PATH
#stm32ai analyze -m ./tmp/retrain_weights_edge.hdf5 --type keras
#stm32ai validate -m ./tmp/retrain_weights_edge.hdf5 --type keras
stm32ai generate -m ./tmp/retrain_weights_edge.hdf5 --type keras \
                 -o ./tmp/cubeai

# transform to .bin
python3 model2bin_server.py \
 -input ./tmp/cubeai/network_data.c \
 -output ./tmp/retrain_weights_edge.bin


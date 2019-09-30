
# CubeAI transform to ./tmp/cubeai/network_data.c 
export X_CUBE_AI_DIR=/home/neusoft/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/4.0.0
export PATH=$X_CUBE_AI_DIR/Utilities/linux:$PATH
#stm32ai analyze -m ./tmp/weights_retrain_edge.hdf5 --type keras
#stm32ai validate -m ./tmp/weights_retrain_edge.hdf5 --type keras
stm32ai generate -m ./tmp/weights_base_edge_1_pre.hdf5 --type keras \
                 -o ./tmp/cubeai

# transform to .bin
python3 model2bin_server.py \
 -input ./tmp/cubeai/network_data.c \
 -output ./tmp/weights_base_edge_1_pre.bin

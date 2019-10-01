python3 train.py \
 -type 1 \
 -pretrain 0 \
 -batch_size 16 \
 -epochs 50 \
 -reload_train 1 \
 -reload_test 1 \
 -data_path /home/neusoft/amy/uInference/data/images_base_all.list \
 -save_model ./tmp/weights_base_edge.hdf5 

python3 convert_model.py \
 -type 1 \
 -model_path ./tmp/weights_base_edge.hdf5 \
 -config_name ./tmp/weights_base_edge.json \
 -weights_name ./tmp/weights_base_edge.h5 \
 -save_name_txt ./tmp/weights_base_edge.txt \
 -save_name_binary ./tmp/weights_base_edge.dat

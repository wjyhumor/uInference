python3 train.py \
 -type 1 \
 -pretrain 0 \
 -batch_size 16 \
 -epochs 50 \
 -reload_train 1 \
 -reload_test 1 \
 -data_path /home/neusoft/amy/uInference/data/images_base_all.list \
 -save_model ./tmp/weights_base_edge_all.hdf5 

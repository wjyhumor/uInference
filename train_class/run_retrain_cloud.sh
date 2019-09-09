python3 retrain.py \
 -type 3 \
 -pretrain 0 \
 -batch_size 16 \
 -epochs 50 \
 -reload_train 1 \
 -reload_test 1 \
 -origin_images ./tmp/images_base_all.list \
 -new_images   /home/neusoft/amy/uInference/data/new_images.list \
 -save_model   ./tmp/weights_retrain_cloud.hdf5 

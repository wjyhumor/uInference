python3 retrain.py \
 -type 3 \
 -pretrain 0 \
 -original_images_list /home/neusoft/amy/uInference/data/water_elec_0516_0625.digits.all \
 -new_images   /home/neusoft/amy/uInference/data/new_images.list \
 -save_model   ./tmp/retrain_weights_cloud.hdf5 \
 -save_builder ./tmp/model_cloud
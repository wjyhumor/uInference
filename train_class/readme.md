#  Explaination of the files
data_base/ : image data for training to have the first model.  
data_new/ : new comming image data for retrain.  
script/ : python code and bash script for retrain.  
  
data_new/new_images.list: a list of new images for retrain of cloud model.  
script/run_retrain_edge.sh: script to retrain edge model.  
script/run_retrain_cloud.sh: script to retrain cloud model.  

# Retrain for the edge units
Run `sh run_retrain_edge.sh` in folder `script/`

# Retrain for the cloud 
Run `sh run_retrain_cloud.sh` in folder `script/`
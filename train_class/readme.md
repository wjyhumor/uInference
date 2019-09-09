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

# Test
`python test.py -test ../data_new/XXXX -model ./tmp/retrain_weights_cloud.hdf5` 
or 
`python test.py -test ../data_new/XXXX -model ./tmp/retrain_weights_edge.hdf5` 

# Deploy cloud model
1.Copy model files to /root/workspace/models/tensorflow/classify/#index (index+=1)
2.Run `docker ps` to get container_id
3.Run `docker stop #container_id`
4.Run cmd as follows to start docker
docker run -p 8500:8500 -p 8501:8501 --mount type=bind,source=/root/workspace/models/tensorflow,target=/models/tensorflow -e MODEL_NAME=yolov3 tensorflow/serving --model_config_file=/models/tensorflow/models.config &
5.Check docker is runing version #index
explorer open http://172.30.1.39:8501/v1/models/classify/metadata


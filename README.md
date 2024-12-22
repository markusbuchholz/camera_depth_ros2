# Reach Alpha Camera Depth ROS 2 


## Run

There are some changes to run model localy.

It is importent to follow instruction and upload the model to ```checkpoints``` folder.

Upload the checkpoints from [here](https://drive.google.com/drive/folders/1vvFFm5wGWGHFtZthLAUwNdOGPQBPVdiF?usp=sharing)

```bash
# Run publisher
cd cam_ws/src/dev_opencv_py/dev_opencv_py 

python3 python3 cam_pub.py


# Run subscriber normal 2D
cd cam_ws/src/dev_opencv_py/dev_opencv_py 

python3 viz_ext_topic_object_follower


# Run depth
cd src/Depth-Anything
python3 ros2_run_local_camera_depth.py

```


## Links
- [Depth-Anything](https://github.com/LiheYoung/Depth-Anything)

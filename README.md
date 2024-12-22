# Reach Alpha Camera Depth ROS 2 

ROS 2 Doceker container to manage RGB camera.

---


There are some changes to run model localy.<br>

It is important to follow instruction and upload the model to ```checkpoints``` folder. <br>

Upload the checkpoints from [here](https://drive.google.com/drive/folders/1vvFFm5wGWGHFtZthLAUwNdOGPQBPVdiF?usp=sharing) <br>


## Note. <br>

Run first camera test.

```bash
# Run publisher
cd cam_ws/src/dev_opencv_py/dev_opencv_py 

python3 python3 test_cam.py
```

## Run

```bash
# Run publisher
cd cam_ws/src/dev_opencv_py/dev_opencv_py 

python3 python3 cam_pub.py
```

```bash
# Run subscriber normal 2D
cd cam_ws/src/dev_opencv_py/dev_opencv_py 

python3 viz_ext_topic_object_follower
```


```bash
# Run depth
cd src/Depth-Anything

python3 ros2_run_local_camera_depth.py

```


## Links
- [Depth-Anything](https://github.com/LiheYoung/Depth-Anything)

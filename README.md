# Reach Alpha Camera Depth ROS 2 

ROS 2 Doceker container to manage RGB camera.

---


There are some changes to run model localy.<br>

It is important to follow instruction and upload the model to ```checkpoints``` folder. <br>

Upload the checkpoints from [here](https://drive.google.com/drive/folders/1vvFFm5wGWGHFtZthLAUwNdOGPQBPVdiF?usp=sharing) <br>


## Note. <br>

Run first camera test.

## Run

1.

```bash
# Run publisher
cd cam_ws/src/dev_opencv_py/dev_opencv_py 

python3 python3 test_cam.py
```

2.

```bash
# Run HSV color checker
cd cam_ws/src/dev_opencv_py/dev_opencv_py 

python3 python3 get_values_from_screen.py
```


3.

```bash
# Run publisher
cd cam_ws/src/dev_opencv_py/dev_opencv_py 

python3 python3 cam_pub.py
```

4.

```bash
# Run subscriber normal 2D
cd cam_ws/src/dev_opencv_py/dev_opencv_py 

python3 viz_ext_topic_object_follower
```

5.

```bash
# Run depth
cd src/Depth-Anything

#visual servoing
python3 ros2_run_local_camera_depth.py

```

6.

```bash
# Run depth
cd src/Depth-Anything

#visual servoing and obstacle
python3 ros2_run_local_camera_depth_for_obs_and_obj.py

```


## Links
- [Depth-Anything](https://github.com/LiheYoung/Depth-Anything)

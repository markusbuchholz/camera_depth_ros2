# Camera Depth ROS 2 

## Introduction

Docker container to manage an RGB camera.<br>
The container also facilitates ROS 2 integration, making it easy to incorporate software with your application.<br>
The concept of 3D position estimation for detecting objects is straightforward.<br>
We detect colors (e.g., red and blue) and estimate the center coordinates (XY).<br>
Then, we check the position of this center (XY) in the 3D model array to retrieve the Z value.<br>

---

There are some changes to run model localy.<br>
It is important to follow instruction and upload the model to ```checkpoints``` folder. <br>
Upload the checkpoints from [here](https://drive.google.com/drive/folders/1vvFFm5wGWGHFtZthLAUwNdOGPQBPVdiF?usp=sharing) <br>

---
Expected results (underwater environment)

![image](https://github.com/user-attachments/assets/1a59a804-9184-4443-b032-504ad9e92691)




## Build


```bash
cd docker

sudo ./build
```


## Run


```bash
cd docker

sudo ./run.sh
```

---

## Run app


## Note. <br>

Run first camera test.


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
Check values on the terminal and adjust your script (e.g.):

```bash
self.white_lower = np.array([0,   15,  112])  
self.white_upper = np.array([179, 78,  193])

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

#for inspection

```/object_localizer```

python3 ros2_object_3d_localizer.py

```


## Links
- [Depth-Anything](https://github.com/LiheYoung/Depth-Anything)

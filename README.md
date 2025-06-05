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

![depth_obj_obs-ezgif](https://github.com/user-attachments/assets/56f09f8d-5081-43b2-9ddf-509a6b99f89b)


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

Notes. <br>

- The camera has to be connected to USB before you start Docker. <br>
- Run first camera test. <br>


```bash
# Run publisher
cd cam_ws/src/dev_opencv_py/dev_opencv_py 

python3 python3 test_cam.py
```
---

## Run app

1.

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

2.

```bash
# Run publisher
cd cam_ws/src/dev_opencv_py/dev_opencv_py 

python3 cam_pub.py
```

3.

```bash
# Run subscriber normal 2D
cd cam_ws/src/dev_opencv_py/dev_opencv_py 

python3 viz_ext_topic_object_follower
```

4.

```bash
# Run depth
cd src/Depth-Anything

#visual servoing
python3 ros2_run_local_camera_depth.py

```

5.

```bash
# Run depth
cd src/Depth-Anything

#visual servoing and obstacle
python3 ros2_run_local_camera_depth_for_obs_and_obj.py

#for inspection
```/object_localizer```

python3 ros2_object_3d_localizer.py

```

## Run teleoperation app

Expected results, <br>
![teleoperation_low](https://github.com/user-attachments/assets/45a94458-1298-459e-9e23-740c43b644a9)

It is required to run other Docker container to simulate the robot arm. <br>

Clone here,<br>

```bash
git clone https://github.com/markusbuchholz/ros2_moveit_manipulators.git
```

Complete command pipeline,


```bash
#terminal 1 
# from ros2_moveit_manipulators Docker container
ros2 launch alpha_bringup_simulation planning_alpha5.launch.py

#terminal 2 
# from ros2_moveit_manipulators Docker container
cd /root/colcon_ws/src/py_alpha_move/py_alpha_move
python3 alpha_ik_controller_sim_const_axis_b.py


#terminal 3
#current Docker
cd /home/devuser/cam_ws/src/dev_opencv_py/dev_opencv_py
python3 cam_pub.py


#terminal 4
#current Docker
cd /home/devuser/src/mediapipe
python3 python3 ros2_xy_gripper_sim_display.py 


```


## RPI

```bash
ssh -t pi@192.168.2.80 "sudo docker exec -it camera_depth_ros2 /bin/bash"
```

---

## Falcon

```bash
ssh -X nx@192.168.2.100
```

## Run Docker 
```bash
cd falcon/camera_depth_ros2/docker
sudo ./run.sh
```

### Run cammera

```bash
cd cam_ws

source install/setup.bash

cd cam_ws/src/dev_opencv_py/dev_opencv_py

python3 cam_jetson.py --camera 6
```

### Run camera GUI on Host

```bash
cd cam_ws/src/dev_opencv_py/dev_opencv_py

python3 ros2_falcon_gui.py
```

### Connect to Docker from Host

```bash
sudo  ssh -t nx@192.168.2.100 "sudo docker exec -it camera_depth_ros2 bash
```
### Run Microstrain sensor

```bash
cd cam_ws

source install/setup.bash

sudo chmod a+rw /dev/ttyACM0

ros2 launch microstrain_inertial_driver microstrain_launch.py

# Check
ros2 topic echo /imu/data
```


## Links
- [Depth-Anything](https://github.com/LiheYoung/Depth-Anything)
- [microstrain_inertial_driver](https://wiki.ros.org/microstrain_inertial_driver)
- [software/ros](https://www.microstrain.com/software/ros)
- [GitHub microstrain](https://github.com/LORD-MicroStrain/microstrain_inertial/tree/ros2)
- [params.yml](https://github.com/LORD-MicroStrain/microstrain_inertial_driver_common/blob/main/config/params.yml)

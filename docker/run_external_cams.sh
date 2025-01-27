#!/usr/bin/env bash

XAUTH=/tmp/.docker.xauth
if [ ! -f "$XAUTH" ]; then
    xauth_list=$(xauth nlist "$DISPLAY")
    xauth_list=$(sed -e 's/^..../ffff/' <<< "$xauth_list")
    if [ -n "$xauth_list" ]; then
        echo "$xauth_list" | xauth -f "$XAUTH" nmerge -
    else
        touch "$XAUTH"
    fi
    chmod a+r "$XAUTH"
fi

local_src="/home/markus/camera_depth_ros2/src"
local_cam_ws="/home/markus/camera_depth_ros2/cam_ws"

VIDEO_DEVICES=""
for video in /dev/video*; do
    # Check if the video device exists to avoid errors
    if [ -e "$video" ]; then
        VIDEO_DEVICES="$VIDEO_DEVICES --device $video:$video"
    fi
done

docker run -it \
    --rm \
    --name camera_depth_ros2 \
    -e DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY="$XAUTH" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v "$XAUTH:$XAUTH" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix" \
    -v "/etc/localtime:/etc/localtime:ro" \
    -v "$local_src:/home/devuser/src/" \
    -v "$local_cam_ws:/home/devuser/cam_ws/" \
    -v "/dev/input:/dev/input" \
    $VIDEO_DEVICES \
    --privileged \
    --security-opt seccomp=unconfined \
    --network host \
    --gpus all \
    camera_depth_ros2:latest

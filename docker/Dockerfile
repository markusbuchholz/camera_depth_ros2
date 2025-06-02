# Markus Buchholz
FROM osrf/ros:humble-desktop AS base

# 1) Remove the old ROS 2 apt source (which has an expired key)
RUN rm -f /etc/apt/sources.list.d/ros2-latest.list

# --- NEW ADDITION: Force a specific, reliable mirror for Ubuntu jammy ---
RUN echo "deb http://de.archive.ubuntu.com/ubuntu/ jammy main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb http://de.archive.ubuntu.com/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://de.archive.ubuntu.com/ubuntu/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://security.ubuntu.com/ubuntu jammy-security main restricted universe multiverse" >> /etc/apt/sources.list

# 1.a) Re-add the official ROS 2 Humble apt repository (and its GPG key)
#      This is required so that apt can find any ros-humble-* packages later.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gnupg2 \
        lsb-release \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
         | apt-key add - \
    && echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu jammy main" \
         > /etc/apt/sources.list.d/ros2-latest.list

# Prevent interactive prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/home/devuser/env/bin:$PATH"

# 2) Now that both Ubuntu and ROS 2 repos are in place, install curl, gnupg2, lsb-release, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        gnupg2 \
        lsb-release && \
    rm -rf /var/lib/apt/lists/*

# 3) Update & install all your essential packages
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        libssl-dev \
        libffi-dev \
        python3 \
        python3-pip \
        python3-venv \
        pkg-config \
        libgtk-3-dev \
        libboost-all-dev \
        patchelf \
        mesa-utils \
        libosmesa6-dev \
        libglfw3-dev \
        libglew-dev \
        xorg-dev \
        xvfb \
        libgl1-mesa-dri \
        libglu1-mesa-dev \
        ffmpeg \
        v4l-utils \
        sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 4) Install additional “extra” OpenGL libraries
RUN apt-get update && \
    apt-get -y --quiet --no-install-recommends install \
        libglvnd0 \
        libgl1 \
        libglx0 \
        libegl1 \
        libxext6 \
        libx11-6 && \
    rm -rf /var/lib/apt/lists/*

# 5) NVIDIA environment variables (unchanged)
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
ENV QT_X11_NO_MITSHM=1

# 6) Create 'devuser' in video,sudo groups
RUN useradd -m -G video,sudo devuser

# 7) Give 'devuser' passwordless sudo
RUN echo "devuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 8) Create a Python venv for 'devuser'
RUN python3 -m venv /home/devuser/env

# 9) Activate venv and upgrade pip
RUN /bin/bash -c "source /home/devuser/env/bin/activate && pip install --upgrade pip"

# 10) Install OpenCV + ArUco into that venv
RUN /bin/bash -c "source /home/devuser/env/bin/activate && pip install opencv-contrib-python"

# 11) Clone your Depth-Anything repo (if you still need it)
RUN git clone https://github.com/LiheYoung/Depth-Anything /home/devuser/Depth-Anything
RUN ls -la /home/devuser/Depth-Anything/
WORKDIR /home/devuser/Depth-Anything

# 12) Install Python requirements inside the venv
RUN /bin/bash -c "source /home/devuser/env/bin/activate && pip install -r requirements.txt"

# 13) Source ROS 2 environment on login
RUN echo "source /opt/ros/humble/setup.bash" >> /home/devuser/.bashrc

# 14) Install MediaPipe in venv
RUN /bin/bash -c "source /home/devuser/env/bin/activate && pip install mediapipe"

# 15) Install microstrain drivers now that ROS2 repo is back
RUN apt-get update && \
    apt-get install -y ros-humble-microstrain-inertial-driver

# 16) Install the RQT plugin for microstrain
RUN apt-get update && \
    apt-get install -y ros-humble-microstrain-inertial-rqt

USER devuser
CMD ["/bin/bash"]

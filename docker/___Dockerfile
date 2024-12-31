#Markus Buchholz
FROM osrf/ros:humble-desktop AS base

# Prevent interactive prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/home/devuser/env/bin:$PATH"

# Update and install essential packages as root
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        curl \
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
        libopencv-dev \
        python3-opencv \
        sudo \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install additional libraries
RUN apt-get update && \
    apt-get -y --quiet --no-install-recommends install \
        libglvnd0 \
        libgl1 \
        libglx0 \
        libegl1 \
        libxext6 \
        libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Set NVIDIA related environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
ENV QT_X11_NO_MITSHM=1

# **Ensure 'devuser' is added to existing 'video' and 'sudo' groups**
RUN useradd -m -G video,sudo devuser

# Grant 'devuser' passwordless sudo privileges
RUN echo "devuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set up a Python virtual environment for 'devuser'
RUN python3 -m venv /home/devuser/env

# Activate the virtual environment and install Python packages
RUN /bin/bash -c "source /home/devuser/env/bin/activate && pip install --upgrade pip && pip install xformers"

# Clone the Depth-Anything repository
RUN git clone https://github.com/LiheYoung/Depth-Anything /home/devuser/Depth-Anything

RUN ls -la /home/devuser/Depth-Anything/

WORKDIR /home/devuser/Depth-Anything

RUN /bin/bash -c "source /home/devuser/env/bin/activate && pip install -r requirements.txt"

# Source ROS 2 setup script for all future shell sessions for devuser
RUN echo "source /opt/ros/humble/setup.bash" >> /home/devuser/.bashrc

USER devuser

CMD ["/bin/bash"]

#!/bin/bash
# Step 1: Install ROS2 Humble Full: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html

# Step 1.1 Set Locale
sudo apt update -y
sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Step 1.2 Setup Sources
sudo apt install software-properties-common
echo -ne '\n' | sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Step 1.3 Install ROS2 Packages
sudo apt update -y
sudo apt upgrade -y
sudo apt install ros-humble-desktop-full -y
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Step 2: Install Mamba: https://robofoundry.medium.com/using-robostack-for-ros2-9bb52ca89c12
cd ~/
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
(echo -e "\nyes"; echo -e "\nyes\n")| bash Mambaforge-$(uname)-$(uname -m).sh
source ~/.bashrc
conda install mamba -c conda-forge -y

# Step 3: Create Mamba environment with nerfstudio: https://docs.nerf.studio/quickstart/installation.html
mamba create -n droid_slam_ros_env python=3.10.12 -y
mamba activate droid_slam_ros_env
python -m pip install --upgrade pip
pip uninstall torch torchvision functorch tinycudann
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install nerfstudio
cd ~/legs_ws/src/droid_slam_ros
python setup.py install

# Step 4: Setup ROS2 Humble with Mamba environment: https://robofoundry.medium.com/using-robostack-for-ros2-9bb52ca89c12 and https://robostack.github.io/GettingStarted.html
conda config --env --add channels conda-forge
conda config --env --add channels robostack
conda config --env --add channels robostack-humble
conda config --env --add channels robostack-experimental
mamba install ros-humble-desktop-full # Will fail
conda config --env --add channels conda-forge
conda config --env --add channels robostack-staging
conda config --env --remove channels defaults
mamba install ros-humble-desktop -y
mamba install ros-humble-desktop-full -y
pip install torch-scatter==2.1.1
pip install matplotlib==3.7.2
pip install matplotlib-inline==0.1.6

# Step 5: Install Realsense2 Package
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
sudo apt-get install ros-humble-realsense2-camera -y

# Step 6: Install Zed SDK: https://www.stereolabs.com/developers/release
cd ~/
wget https://download.stereolabs.com/zedsdk/4.1/cu121/ubuntu22
chmod +x ubuntu22
./ubuntu22 # Install everything with default settings (say yes) except for the optimize models because we didn't want to wait for it to take multiple hours

# Step 7: Install Zed ROS2 Package: https://www.stereolabs.com/docs/ros2
cd ~/legs_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --cmake-args=-DCMAKE_BUILD_TYPE=Release

# Step 8: Install OpenCV and OpenCV Contrib: https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html
cd ~/
sudo apt update && sudo apt install -y cmake g++ wget unzip
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
unzip opencv.zip
unzip opencv_contrib.zip
mkdir -p build && cd build
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x
cmake --build .

# Step 9: Install ROS2 Image Transport
sudo apt-get install ros-humble-image-transport

# Step 10: Build workspace
cd ~/legs_ws
colcon build
. install/setup.bash
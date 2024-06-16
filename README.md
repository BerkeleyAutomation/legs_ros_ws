# LEGS
Hi! Welcome to the ROS2 workspace setup for LEGS. This README details how to setup the ROS2 environment with Droid SLAM and how to collect data and get Image-Pose correspondances for our hardware setup with 2 left and right facing Zed2s and a front facing Realsense D455. However, hopefully things should be clear enough to adapt to whatever depth camera hardware you are using. Hopefully, you can read more about this in our paper (IROS 2024 hopefully ;)

## Prerequesites
Computer with Ubuntu 22.04 and GPU with at least 20 GB RAM (this is where we have observed the max spike occurs). If you have a smaller GPU, things should still work, you just have to downsample your images, which could mitigate quality but it shouldn't significantly make things worse.

## Installation (Bash Script)

Clone the repo:
```
cd ~/
mkdir legs_ws
cd legs_ws
git clone --recurse-submodules https://github.com/BerkeleyAutomation/legs_ros_ws.git src
```

Run the setup bash script
```
cd ~/legs_ws/src
bash env_setup.bash
```

## Installation (Step by Step)

### Step 1: Install ROS2 Humble Full: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html
```
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
```

### Step 2: Install Mamba: https://robofoundry.medium.com/using-robostack-for-ros2-9bb52ca89c12
```
cd ~/
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
(echo -e "\nyes"; echo -e "\nyes\n")| bash Mambaforge-$(uname)-$(uname -m).sh
source ~/.bashrc
conda install mamba -c conda-forge -y
```

### Step 3: Create Mamba environment with nerfstudio: https://docs.nerf.studio/quickstart/installation.html
```
mamba create -n droid_slam_ros_env python=3.10.12 -y
mamba activate droid_slam_ros_env
python -m pip install --upgrade pip
pip uninstall torch torchvision functorch tinycudann
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
pip install setuptools==69.5.1 # ImportError: cannot import name packaging from pkg_resources
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install nerfstudio
```

### Step 4: Setup ROS2 Humble with Mamba environment: https://robofoundry.medium.com/using-robostack-for-ros2-9bb52ca89c12 and https://robostack.github.io/GettingStarted.html
```
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
```

### Step 5: Install Realsense2 Package
```
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
sudo apt-get install ros-humble-realsense2-camera -y
```

### Step 6: Install Zed SDK: https://www.stereolabs.com/developers/release
```
cd ~/
wget https://download.stereolabs.com/zedsdk/4.1/cu121/ubuntu22
chmod +x ubuntu22
./ubuntu22 # Install everything with default settings (say yes) except for the optimize models because we didn't want to wait for it to take multiple hours
```

### Step 7: Install Zed ROS2 Package: https://www.stereolabs.com/docs/ros2
```
cd ~/legs_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --cmake-args=-DCMAKE_BUILD_TYPE=Release
```

### Step 8: Install OpenCV and OpenCV Contrib: https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html
```
cd ~/
sudo apt update && sudo apt install -y cmake g++ wget unzip
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
unzip opencv.zip
unzip opencv_contrib.zip
mkdir -p build && cd build
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x
cmake --build .
```
### Step 9: Install ROS2 Image Transport
```
sudo apt-get install ros-humble-image-transport
```

### Step 10: Build workspace
```
cd ~/legs_ws
colcon build
. install/setup.bash
```

## Run Project

### Run Realsense D455
```
cd ~/legs_ws
colcon build
. install/setup.bash
ros2 launch camera_bringup realsense_d455.launch.py
```

To verify things are working, you can open Rviz and view topics /camera/color/image_raw and /camera/depth/image_rect_raw.

### Run Left Zed (with TRI Stereo Model: https://arxiv.org/pdf/2109.11644)

Terminal 1
```
cd ~/legs_ws
colcon build
. install/setup.bash
ros2 launch zed_wrapper left_zed_camera.launch.py
```
Terminal 2
```
cd ~/legs_ws
colcon build
. install/setup.bash
ros2 run tri_model_ros_wrapper left_tri_model_node.py
```
To verify things are working, you can open Rviz and view topics /tri_left_zed_cropped and /tri_left_zed_depth. If you don't have access to the super secret, level 13 classified clearance stereo model, you should be able to use topics /left_zed/zed_node/left/image_rect_color and /left_zed/zed_node/depth/depth_registered.

### Run Right Zed (with TRI Stereo Model: https://arxiv.org/pdf/2109.11644)

Terminal 1
```
cd ~/legs_ws
colcon build
. install/setup.bash
ros2 launch zed_wrapper right_zed_camera.launch.py
```
Terminal 2
```
cd ~/legs_ws
colcon build
. install/setup.bash
ros2 run tri_model_ros_wrapper right_tri_model_node.py
```
To verify things are working, you can open Rviz and view topics /tri_right_zed_cropped and /tri_right_zed_depth. If you don't have access to the super secret, level 13 classified clearance stereo model, you should be able to use topics /right_zed/zed_node/left/image_rect_color and /right_zed/zed_node/depth/depth_registered.

### Run all cameras synced up together
Run the Realsense D455, left Zed, and right Zed as explained above. Then, also open a terminal and run the following. This node will make sure all the timestamps are synced up, and then publish all the images at 5 Hz to reduce the strain on network bandwidth. This frame rate can be changed within the node.
```
cd ~/legs_ws
colcon build
. install/setup.bash
ros2 run camera_bringup synced_image_transport_node
```
To verify things are working, see if the topics are being published at the proper frequency. It may be tough to view the images on Rviz traditionally, but you can run the uncompressed_zed_realsense_image_transport_node node to visualize the compressed images.
```
cd ~/legs_ws
colcon build
. install/setup.bash
ros2 run camera_bringup uncompressed_zed_realsense_image_transport_node
```

### Save synced camera data to onboard computer
Run the Realsense D455, left Zed, and right Zed as explained above. Then, also open a terminal and run the following. This node will save all the RGB and depth images on the computer as pngs and numpy files.
```
cd ~/legs_ws
colcon build
. install/setup.bash
ros2 run camera_bringup auto_splat_sync_collector_node.py
```
To verify things are working, check to see if all images are saved in the folder called sync_images.

### Convert synced camera data to ROS2 bag
So let's say you made a bunch of RGB and depth folders from the auto_splat_sync_collector_node. If we want to publish these files to ROS2 topics and make a ROS2 bag out of it, run the following 2 terminals. Before you run the convert_d455_2_zed_folder_to_bag.py script, change the d455_folder variable to the desired filepath.

Terminal 1
```
ros2 bag record -o data_bag -a
```

Terminal 2
```
cd ~/legs_ws
colcon build
. install/setup.bash
ros2 run droid_slam_ros convert_d455_2_zed_folder_to_bag.py
```
To verify things are working, run the view_multi_compressed_bag.py node which should view all the images.
```
cd ~/legs_ws
colcon build
. install/setup.bash
ros2 run droid_slam_ros view_multi_compressed_bag.py
```

### Run DROID-SLAM with spacebar bundle adjust
Either play a data bag or run the cameras all synced up together and then run the following on 2 terminals. 

Terminal 1
```
cd ~/legs_ws
colcon build
. install/setup.bash
ros2 run droid_slam_ros space_bundle_adjust_node.py
```

Terminal 2
```
cd ~/legs_ws
colcon build
. install/setup.bash
ros2 run droid_slam_ros multi_ptcloud_d455_2zed_global_real_droid_subscriber_node_prime.py
```
To verify things are working, you should be able to open a Viser window in the browser and see the SLAM happening in realtime. Everytime you want to run the global bundle adjustment, make sure the robot isn't moving and hit the spacebar in the bundle adjust node terminal.

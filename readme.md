# Perception Workspace 
## (Containing 'perception' ROS2 Foxy pkg)


### ROS Dependencies (ZED2)
```
ros-foxy-xacro
ros-foxy-robot-localization
ros-foxy-nmea-msgs
ros-foxy-ament-cmake-clang-format
ros-foxy-vision-msgs
```


### CUDA 12.x Setup
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu


### Building workspace
```
chmod +x ./build_packages.sh
./build_packages.sh
```

### in workspace clone
```
git clone https://github.com/stereolabs/zed-ros2-wrapper
git clone https://github.com/stereolabs/zed-ros2-examples
git clone https://github.com/stereolabs/zed-ros2-interfaces
git clone https://github.com/ros-perception/image_common

```
#!/bin/bash
set -e

cd src/
git clone https://github.com/stereolabs/zed-ros2-interfaces.git
cd ../
sudo apt update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --cmake-args=-DCMAKE_BUILD_TYPE=Release
echo source $(pwd)/install/local_setup.bash >> ~/.bashrc
source ~/.bashrc
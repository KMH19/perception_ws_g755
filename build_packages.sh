#!/bin/bash
set -e

source "/opt/ros/foxy/setup.bash"
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --cmake-args=-DCMAKE_BUILD_TYPE=Release
source ~/.bashrc

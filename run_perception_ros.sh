#!/bin/bash
set -e

# setup ros2 environment
source "/opt/ros/foxy/setup.bash"
source "install/setup.bash"

export CUDA_LAUNCH_BLOCKING=1
ros2 run perception perception_node
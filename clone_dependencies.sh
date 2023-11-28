#!/bin/bash
set -e

cd src/

git clone https://github.com/stereolabs/zed-ros2-interfaces.git
git clone  --recursive https://github.com/stereolabs/zed-ros2-wrapper.git
git clone https://github.com/stereolabs/zed-ros2-examples.git
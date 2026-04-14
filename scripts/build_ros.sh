#!/bin/bash

# Build the main project
cd /workspace/repo
./scripts/build.sh

# Build the ROS package
cd /root/catkin_ws
catkin build diskchungs_ros --cmake-args -DCMAKE_BUILD_TYPE=Release

# Source the setup file
source /root/catkin_ws/devel/setup.bash

cd /workspace/repo
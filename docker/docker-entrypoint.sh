#!/bin/bash

# Set up all environment variables directly in this script
echo "Loading environment variables..."

# OpenCV
# export LD_PRELOAD=/workspace/third_party/install/opencv/lib/libopencv_core.so.410:/workspace/third_party/install/opencv/lib/libopencv_imgproc.so.410:/workspace/third_party/install/opencv/lib/libopencv_imgcodecs.so.410:/workspace/third_party/install/opencv/lib/libopencv_videoio.so.410:/workspace/third_party/install/opencv/lib/libopencv_highgui.so.410
export OPENCV_PATH=/workspace/third_party/opencv
export OpenCV_DIR=/workspace/third_party/install/opencv/lib/cmake/opencv4
export LD_LIBRARY_PATH=/workspace/third_party/install/opencv/lib:$LD_LIBRARY_PATH

# ROS
source /opt/ros/noetic/setup.bash

# libtorch
export LD_LIBRARY_PATH=/opt/ros/noetic/lib:/workspace/third_party/libtorch/lib:$LD_LIBRARY_PATH

echo "Environment setup complete!"

# Setup ROS workspace if needed
if [ -d "/workspace/repo/ros_wrapper" ]; then
  echo "Setting up ROS workspace..."
  cd ~/catkin_ws/src
  # Create symlink if it doesn't exist
  if [ ! -L "ros_wrapper" ]; then
    ln -sf /workspace/repo/ros_wrapper .
  fi
fi

cd /workspace/repo

# Download mono model only if it doesn't exist
MODEL_PATH="/workspace/repo/models/depth_anything_v2_vitl.onnx"
if [ ! -f "$MODEL_PATH" ]; then
  echo "Downloading mono model..."
  wget -P /workspace/repo/models https://huggingface.co/yuvraj108c/Depth-Anything-2-Onnx/resolve/main/depth_anything_v2_vitl.onnx
else
  echo "Mono model already exists, skipping download."
fi

# Print welcome message
echo "==============================================="
echo "3DGS SLAM Docker Environment"
echo "==============================================="
echo "- OpenCV is built and installed"
echo "- PyTorch and libtorch are ready to use"
echo "- ROS Noetic is configured"
echo "- Environment variables are set"
echo ""
echo "Code is available at /workspace/repo"
echo "Build DiskChunGS with ./scripts/build.sh"
echo "==============================================="

# Execute the command passed to docker run
exec "$@"
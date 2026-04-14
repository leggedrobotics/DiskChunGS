#!/bin/bash

# Get repo root (parent of scripts/)
workdir=$( cd -- "$(dirname "$0")/.." >/dev/null 2>&1 ; pwd -P )
echo "Cleaning build in: $workdir"
cd "$workdir"

# Clean main project build
echo "Cleaning main project build directory..."
rm -rf build/

# Clean DBoW2
echo "Cleaning DBoW2..."
rm -rf third_party/ORB-SLAM3/Thirdparty/DBoW2/build/

# Clean g2o
echo "Cleaning g2o..."
rm -rf third_party/ORB-SLAM3/Thirdparty/g2o/build/

# Clean Sophus
echo "Cleaning Sophus..."
rm -rf third_party/ORB-SLAM3/Thirdparty/Sophus/build/

# Clean ORB-SLAM3
echo "Cleaning ORB-SLAM3..."
rm -rf third_party/ORB-SLAM3/build/
rm -f third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt

echo "Clean completed!"
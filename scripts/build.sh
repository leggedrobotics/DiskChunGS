#!/bin/bash
set -e  # Exit on any error

# Get repo root (parent of scripts/)
workdir=$( cd -- "$(dirname "$0")/.." >/dev/null 2>&1 ; pwd -P )
echo "Working directory: $workdir"
cd "$workdir"

export OPENCV_PATH=/workspace/third_party/opencv
export OpenCV_DIR=/workspace/third_party/install/opencv/lib/cmake/opencv4
export LD_LIBRARY_PATH=/workspace/third_party/install/opencv/lib:$LD_LIBRARY_PATH

# Set compiler flags
export CMAKE_EXPORT_COMPILE_COMMANDS=ON

# --- Build ORB-SLAM3 dependencies ---
echo "Building ORB-SLAM3 dependencies..."

# DBoW2
echo "Building DBoW2..."
cmake -B third_party/ORB-SLAM3/Thirdparty/DBoW2/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DOpenCV_DIR=/workspace/third_party/install/opencv/lib/cmake/opencv4 \
      third_party/ORB-SLAM3/Thirdparty/DBoW2
cmake --build third_party/ORB-SLAM3/Thirdparty/DBoW2/build

# g2o
echo "Building g2o..."
cmake -B third_party/ORB-SLAM3/Thirdparty/g2o/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      third_party/ORB-SLAM3/Thirdparty/g2o
cmake --build third_party/ORB-SLAM3/Thirdparty/g2o/build

# Sophus
echo "Building Sophus..."
cmake -B third_party/ORB-SLAM3/Thirdparty/Sophus/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      third_party/ORB-SLAM3/Thirdparty/Sophus
cmake --build third_party/ORB-SLAM3/Thirdparty/Sophus/build

# Uncompress vocabulary if needed
if [ -f "third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt.tar.gz" ] && [ ! -f "third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt" ]; then
    echo "Uncompressing vocabulary..."
    tar -xf third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt.tar.gz \
        -C third_party/ORB-SLAM3/Vocabulary
fi

# Build ORB-SLAM3
echo "Building ORB-SLAM3..."
cmake -B third_party/ORB-SLAM3/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations" \
      -DOpenCV_DIR=/workspace/third_party/install/opencv/lib/cmake/opencv4 \
      third_party/ORB-SLAM3
cmake --build third_party/ORB-SLAM3/build

# Update PATH for ORB-SLAM3 library
export LD_LIBRARY_PATH=$workdir/third_party/ORB-SLAM3/lib:$LD_LIBRARY_PATH
echo "ORB-SLAM3 built successfully!"

# --- Build main application ---
echo "Building DiskChunGS application..."
ARCH=$(uname -m)

if [ "$ARCH" = "x86_64" ]; then
  CUDA_ARCH="86"
  CXX_FLAGS="-O3 -march=native -mtune=native -ffast-math -fopenmp -DWITH_CUDA"
  CUDA_FLAGS="-O3 -use_fast_math"
elif [ "$ARCH" = "aarch64" ]; then
  CUDA_ARCH="87"
  CXX_FLAGS="-O3 -march=native -mtune=native -fopenmp -DWITH_CUDA"
  CUDA_FLAGS="-O3"
else
  echo "Unknown architecture: $ARCH"
  exit 1
fi

cmake -B build -G Ninja \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
 -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
 -DTorch_DIR=/workspace/third_party/libtorch/share/cmake/Torch \
 -DOpenCV_DIR=/workspace/third_party/install/opencv/lib/cmake/opencv4 \
 -DCMAKE_CXX_FLAGS="$CXX_FLAGS" \
 -DCMAKE_CUDA_FLAGS="$CUDA_FLAGS" \
 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++

cmake --build build -j16 # Reduce if causes OOM

echo "✓ Build completed successfully!"
echo "----------------------------------------"
echo "You can run the application with:"
echo "  ./bin/application_name"
echo "----------------------------------------"
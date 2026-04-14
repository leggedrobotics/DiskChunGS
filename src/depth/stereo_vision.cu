/**
 * This file is part of DiskChunGS, modified from CaRtGS/Photo-SLAM.
 *
 * Original Copyright (C) 2023-2024 Longwei Li, Hui Cheng (Photo-SLAM)
 * Modified Copyright (C) 2024 Dapeng Feng (CaRtGS)
 * Modified Copyright (C) 2025 Robotic Systems Lab, ETH Zurich (DiskChunGS)
 *
 * This software is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * See the GNU General Public License for more details:
 * <http://www.gnu.org/licenses/>.
 */

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <fstream>
#include <iostream>
#include <numeric>

#include "depth/stereo_vision.h"
namespace cg = cooperative_groups;

#include "cuda_rasterizer/operate_points.h"
#include "cuda_rasterizer/stereo_vision.h"

__global__ void reproject_depths_pinhole(const int P,
                                         const int width,
                                         const float fx,
                                         const float fy,
                                         const float cx,
                                         const float cy,
                                         const float* __restrict__ depths,
                                         const bool* __restrict__ mask,
                                         float* __restrict__ points) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P || !mask[idx]) return;

  int v = idx / width;
  int u = idx - v * width;
  float depth = depths[idx];

  float3 point = reproject_depth_pinhole(u, v, depth, fx, fy, cx, cy);
  insert_point_to_pcd(idx, point, points);
}

torch::Tensor reprojectDepthPinhole(torch::Tensor& depth,
                                    torch::Tensor& mask,
                                    std::vector<float>& intr,
                                    int width) {
  if (depth.ndimension() != 1) {
    AT_ERROR("points must have dimensions (num_points)");
  }

  const int P = depth.size(0);
  torch::Tensor points;

  if (P != 0) {
    points = torch::zeros({P, 3}, depth.options());

    float fx = intr[0];
    float fy = intr[1];
    float cx = intr[2];
    float cy = intr[3];

    reproject_depths_pinhole<<<(P + 255) / 256, 256>>>(
        P, width, fx, fy, cx, cy, depth.contiguous().data_ptr<float>(),
        mask.contiguous().data_ptr<bool>(),
        points.contiguous().data_ptr<float>());
  }

  return points;
}
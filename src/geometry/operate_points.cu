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

#include "geometry/operate_points.h"
namespace cg = cooperative_groups;

#include "cuda_rasterizer/operate_points.h"

__global__ void transform_points(const int P,
                                 const float* __restrict__ orig_points,
                                 const float* __restrict__ transformmatrix,
                                 float* __restrict__ trans_points) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P) return;

  float3 p_trans = transform_point(idx, orig_points, transformmatrix);
  insert_point_to_pcd(idx, p_trans, trans_points);
}

__global__ void scale_and_transform_points(
    const int P,
    const float scale,
    const float* __restrict__ orig_points,
    const float* __restrict__ orig_rots,
    const float* __restrict__ transformmatrix,
    const bool* __restrict__ mask,
    float* __restrict__ trans_points,
    float* __restrict__ trans_rots) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P || !mask[idx]) return;

  float3 p_trans =
      scale_and_transform_point(idx, scale, orig_points, transformmatrix);
  insert_point_to_pcd(idx, p_trans, trans_points);

  float4 rot_trans =
      transfrom_quaternion_using_matrix(idx, orig_rots, transformmatrix);
  insert_rot_to_rots(idx, rot_trans, trans_rots);
}

void transformPoints(torch::Tensor& points, torch::Tensor& transformmatrix) {
  if (points.ndimension() != 2 || points.size(1) != 3) {
    AT_ERROR("points must have dimensions (num_points, 3)");
  }

  const int P = points.size(0);
  torch::Tensor transformed_points = torch::zeros_like(points);

  if (P != 0) {
    transform_points<<<(P + 255) / 256, 256>>>(
        P, points.contiguous().data_ptr<float>(),
        transformmatrix.contiguous().data_ptr<float>(),
        transformed_points.contiguous().data_ptr<float>());

    points = transformed_points;
  }
}

void scaleAndTransformThenMarkVisiblePoints(
    torch::Tensor& points,
    torch::Tensor& rots,
    torch::Tensor& point_transformed_mask,
    torch::Tensor& point_unstable_mask,
    const torch::Tensor& transformmatrix,
    torch::Tensor& viewmatrix,
    torch::Tensor& projmatrix,
    int& num_transformed,
    const float scale) {
  if (points.ndimension() != 2 || points.size(1) != 3) {
    TORCH_CHECK(false, "points must have dimensions (num_points, 3)");
  }

  torch::Tensor present = markVisible(points, viewmatrix, projmatrix);

  // std::cout << "[DEBUG] Visible points: " << present.sum().item<int>()
  //           << " out of " << present.size(0) << std::endl;

  auto num_points = present.size(0);
  if (point_transformed_mask.size(0) != num_points ||
      point_unstable_mask.size(0) != num_points) {
    // std::cout << "[DEBUG] point_transformed_mask size: "
    //           << point_transformed_mask.size(0)
    //           << ", point_unstable_mask size: " <<
    //           point_unstable_mask.size(0)
    //           << std::endl;
    TORCH_CHECK(false, "points_mask must have dimensions (num_points)");
  }

  // std::cout << "[DEBUG] Points unstable mask true count: "
  //           << point_unstable_mask.sum().item<int>() << std::endl;
  torch::Tensor final_mask = torch::logical_and(
      torch::logical_not(point_transformed_mask), point_unstable_mask);
  final_mask = torch::logical_and(final_mask, present);

  // std::cout << "[DEBUG] Not transformed mask true count: "
  //           << torch::logical_not(point_transformed_mask).sum().item<int>()
  //           << std::endl;

  // std::cout << "[DEBUG] Final mask (intersection) true count: "
  //           << final_mask.sum().item<int>() << std::endl;

  num_transformed += final_mask.sum().item<int>();
  const int P = points.size(0);

  if (P != 0) {
    torch::Tensor transformed_points = torch::zeros_like(points);
    torch::Tensor transformed_rots = torch::zeros_like(rots);

    scale_and_transform_points<<<(P + 255) / 256, 256>>>(
        P, scale, points.contiguous().data_ptr<float>(),
        rots.contiguous().data_ptr<float>(),
        transformmatrix.contiguous().data_ptr<float>(),
        final_mask.contiguous().data_ptr<bool>(),
        transformed_points.contiguous().data_ptr<float>(),
        transformed_rots.contiguous()
            .data_ptr<float>());  // FIXED: Output scales

    points.index_put_({final_mask}, transformed_points.index({final_mask}));
    rots.index_put_({final_mask}, transformed_rots.index({final_mask}));
    point_transformed_mask.index_put_(
        {final_mask}, torch::full({P}, true, point_transformed_mask.options())
                          .index({final_mask}));
  }
}

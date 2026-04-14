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

#pragma once

#include <torch/torch.h>

#include "cuda_rasterizer/rasterize_points.h"

void transformPoints(torch::Tensor& points, torch::Tensor& transformmatrix);

void scaleAndTransformThenMarkVisiblePoints(
    torch::Tensor& points,
    torch::Tensor& rots,
    torch::Tensor& point_not_transformed_mask,
    torch::Tensor& point_unstable_mask,
    const torch::Tensor& transformmatrix,
    torch::Tensor& viewmatrix,
    torch::Tensor& projmatrix,
    int& num_transformed,
    const float scale = 1.0f);

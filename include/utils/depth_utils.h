/**
 * This file is part of DiskChunGS.
 *
 * Copyright (C) 2025 Robotic Systems Lab, ETH Zurich (DiskChunGS)
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

namespace depth_utils {

/**
 * @brief Compute depth confidence using Sobel edge detection
 *
 * Computes confidence map from depth image by detecting edges using Sobel
 * filters. Higher confidence is assigned to regions with lower edge magnitude.
 * This is useful for weighting depth information based on local smoothness.
 *
 * @param depth_image Input depth tensor (should be 4D: [B, C, H, W])
 * @return Confidence tensor with values in [0, 1], same shape as input
 */
inline torch::Tensor computeDepthConfidence(const torch::Tensor& depth_image) {
  constexpr float EDGE_VARIANCE = 0.2f;

  // Initialize Sobel kernels for edge detection
  torch::Tensor sobel_x = torch::tensor(
      {{{{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}}},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  torch::Tensor sobel_y = torch::tensor(
      {{{{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}}}},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  // Compute gradients using Sobel filters
  torch::Tensor grad_x = torch::nn::functional::conv2d(
      depth_image, sobel_x,
      torch::nn::functional::Conv2dFuncOptions().padding(1));

  torch::Tensor grad_y = torch::nn::functional::conv2d(
      depth_image, sobel_y,
      torch::nn::functional::Conv2dFuncOptions().padding(1));

  // Compute edge magnitude and confidence
  torch::Tensor edges = torch::cat({grad_x, grad_y}, 0);
  torch::Tensor edges_sq_norm = (edges.pow(2)).sum(0, true);

  return torch::exp(-edges_sq_norm / EDGE_VARIANCE);
}

}  // namespace depth_utils

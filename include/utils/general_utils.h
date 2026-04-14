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

#include "gaussian_splatting/utils/general_utils.h"

#include <torch/torch.h>

namespace general_utils {

inline torch::Tensor rgb2gray(const torch::Tensor& rgb,
                              const float gamma = 2.2) {
  auto r = rgb[0];
  auto g = rgb[1];
  auto b = rgb[2];

  auto gray =
      torch::pow(0.2973 * torch::pow(r, gamma) + 0.6274 * torch::pow(g, gamma) +
                     0.0753 * torch::pow(b, gamma),
                 1 / gamma);
  return gray;
}

}  // namespace general_utils

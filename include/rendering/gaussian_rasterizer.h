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
 * See <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "gaussian_splatting/rendering/gaussian_rasterizer.h"

/**
 * @brief Adam optimizer variant for sparse Gaussian parameter updates.
 */
class SparseGaussianAdam : public torch::optim::Adam {
 public:
  explicit SparseGaussianAdam(std::vector<torch::Tensor> parameters,
                              const torch::optim::AdamOptions& options)
      : torch::optim::Adam(parameters, options) {}
};

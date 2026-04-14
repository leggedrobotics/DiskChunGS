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

#include "gaussian_splatting/utils/loss_utils.h"

#include <torch/torch.h>

namespace loss_utils {

inline torch::Tensor l1_depth_loss(torch::Tensor &network_output,
                                   torch::Tensor &gt) {
  torch::Tensor loss = torch::abs(network_output - gt);
  loss = loss.masked_fill(gt == 0, 0);
  return loss.mean();
}

inline torch::Tensor smooth_l1_depth_loss(torch::Tensor &network_output,
                                          torch::Tensor &gt,
                                          float beta = 1.0) {
  torch::Tensor diff = torch::abs(network_output - gt);
  torch::Tensor loss =
      torch::where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta);
  loss = loss.masked_fill(gt == 0, 0);
  return loss.mean();
}

inline torch::Tensor scale_invariant_depth_loss(torch::Tensor &network_output,
                                                torch::Tensor &gt_relative) {
  // Create mask for valid depth values
  auto mask = (gt_relative > 0) & (network_output > 0);

  if (mask.sum().item<int>() == 0) {
    return torch::zeros({}, network_output.options());
  }

  auto pred_masked = network_output.masked_select(mask);
  auto gt_masked = gt_relative.masked_select(mask);

  // Take log to make it scale-invariant
  auto log_pred = torch::log(pred_masked + 1e-8);
  auto log_gt = torch::log(gt_masked + 1e-8);

  auto diff = log_pred - log_gt;
  auto loss = diff.pow(2).mean() - 0.5 * diff.mean().pow(2);

  return loss;
}

inline torch::Tensor smooth_l1_loss(torch::Tensor &network_output,
                                    torch::Tensor &gt,
                                    const float beta = 1.0f) {
  auto diff = torch::abs(network_output - gt);
  auto loss =
      torch::where(diff < beta, 0.5 * diff.pow(2) / beta, diff - 0.5 * beta);

  return loss.mean();
}

inline torch::Tensor total_variation_loss(torch::Tensor &img) {
  auto diff_i = torch::abs(img.slice(2, 1, img.size(2)) -
                           img.slice(2, 0, img.size(2) - 1));
  auto diff_j = torch::abs(img.slice(1, 1, img.size(1)) -
                           img.slice(1, 0, img.size(1) - 1));

  return diff_i.mean() + diff_j.mean();
}

}  // namespace loss_utils

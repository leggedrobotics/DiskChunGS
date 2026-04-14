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

#include <torch/torch.h>

#include <memory>
#include <utility>
#include <vector>

#include "gaussian_splatting/depth/guided_mvs.h"
#include "scene/gaussian_keyframe.h"

/**
 * @brief Keyframe-aware guided MVS depth estimation.
 *
 * Extracts feature maps, poses, intrinsics, and depth priors from
 * GaussianKeyframe objects for multi-view stereo estimation.
 */
class GuidedMVS : public GuidedMVSBase {
 public:
  GuidedMVS(int num_prev_keyframes,
            int num_depth_candidates = 16,
            float inverse_depth_range = 2e-1f)
      : GuidedMVSBase(num_prev_keyframes, num_depth_candidates,
                       inverse_depth_range) {}

  /**
   * @brief Perform guided MVS depth estimation using keyframe data.
   */
  std::pair<torch::Tensor, torch::Tensor> operator()(
      const torch::Tensor& uv,
      const std::shared_ptr<GaussianKeyframe> refKeyframe,
      const std::vector<std::shared_ptr<GaussianKeyframe>>& keyframes);
};

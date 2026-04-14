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
#include <tuple>

#include "gaussian_splatting/rendering/gaussian_renderer.h"
#include "model/gaussian_model.h"
#include "scene/gaussian_keyframe.h"
#include "gaussian_splatting/scene/gaussian_parameters.h"

/**
 * @brief DiskChunGS renderer with keyframe-aware exposure correction.
 */
class GaussianRenderer {
 public:
  static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  render(std::shared_ptr<GaussianModel> model,
         const torch::Tensor& visible_gaussian_mask,
         std::shared_ptr<GaussianKeyframe> viewpoint_camera,
         int image_height,
         int image_width,
         GaussianPipelineParams& pipe,
         torch::Tensor& bg_color,
         torch::Tensor& override_color,
         float scaling_modifier,
         bool use_override_color,
         float FoVx,
         float FoVy,
         torch::Tensor& world_view_transform,
         torch::Tensor& projection_matrix);
};

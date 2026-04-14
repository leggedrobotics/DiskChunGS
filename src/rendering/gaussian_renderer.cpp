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

#include "rendering/gaussian_renderer.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GaussianRenderer::render(std::shared_ptr<GaussianModel> model,
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
                         torch::Tensor& projection_matrix) {
  torch::Tensor camera_center = viewpoint_camera->getCenter();

  auto [rendered_depth, rendered_image, radii, mainGaussID] =
      GaussianRendererBase::render(
          model.get(), visible_gaussian_mask, camera_center, image_height,
          image_width, pipe, bg_color, override_color, scaling_modifier,
          use_override_color, FoVx, FoVy, world_view_transform,
          projection_matrix);

  rendered_image = viewpoint_camera->applyExposureTransform(rendered_image);

  return std::make_tuple(rendered_depth, rendered_image, radii, mainGaussID);
}

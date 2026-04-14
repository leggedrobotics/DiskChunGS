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

#include "depth/guided_mvs.h"

#include <torch/torch.h>

std::pair<torch::Tensor, torch::Tensor> GuidedMVS::operator()(
    const torch::Tensor& uv,
    const std::shared_ptr<GaussianKeyframe> refKeyframe,
    const std::vector<std::shared_ptr<GaussianKeyframe>>& keyframes) {
  // Compute relative poses: other keyframes to reference frame
  std::vector<torch::Tensor> other2ref_list;
  for (const auto& keyframe : keyframes) {
    auto rel_pose = torch::matmul(keyframe->getRT(),
                                  torch::linalg::inv(refKeyframe->getRT()));
    other2ref_list.push_back(rel_pose.slice(0, 0, 3).slice(1, 0, 4));
  }
  auto relative_poses = torch::stack(other2ref_list, 0).contiguous().cuda();

  // Gather feature maps
  auto ref_feature_map = refKeyframe->feature_map_.contiguous().cuda();
  std::vector<torch::Tensor> feat_list;
  for (const auto& keyframe : keyframes) {
    feat_list.push_back(keyframe->feature_map_.cuda().contiguous());
  }
  auto other_feature_maps = torch::stack(feat_list, 0);

  // Extract intrinsics
  auto intrinsics = torch::tensor({refKeyframe->intr_[0], refKeyframe->intr_[2],
                                   refKeyframe->intr_[3]})
                        .contiguous()
                        .cuda();

  // Get monocular inverse depth prior
  auto mono_idepth = refKeyframe->gaus_pyramid_inv_depth_image_[0]
                         .unsqueeze(0)
                         .unsqueeze(0)
                         .contiguous()
                         .cuda();

  int image_height = static_cast<int>(
      refKeyframe->gaus_pyramid_original_image_[0].size(1));
  int image_width = static_cast<int>(
      refKeyframe->gaus_pyramid_original_image_[0].size(2));

  return estimate(uv, ref_feature_map, other_feature_maps, relative_poses,
                  intrinsics, mono_idepth, image_height, image_width);
}

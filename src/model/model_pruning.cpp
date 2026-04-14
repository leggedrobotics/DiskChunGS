/**
 * This file is part of DiskChunGS.
 *
 * Copyright (C) 2023-2024 Longwei Li, Hui Cheng (Photo-SLAM)
 * Copyright (C) 2024 Dapeng Feng (CaRtGS)
 * Copyright (C) 2025 Robotic Systems Lab, ETH Zurich (DiskChunGS)
 *
 * This software is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * See <http://www.gnu.org/licenses/>.
 */

#include "model/gaussian_model.h"
#include "rendering/gaussian_rasterizer.h"

void GaussianModel::resetOpacityForMask(const torch::Tensor& gaussian_mask) {
  torch::NoGradGuard no_grad;

  int num_reset = torch::sum(gaussian_mask).item<int>();
  std::cout << "[Opacity Reset] Resetting opacity for " << num_reset
            << " gaussians" << std::endl;

  torch::Tensor current_opacity_activated = getOpacityActivation();

  // min(current, 0.05) for masked gaussians, then convert back to logit space
  torch::Tensor target_opacity =
      torch::min(current_opacity_activated,
                 torch::ones_like(current_opacity_activated) * 0.05f);
  torch::Tensor new_opacity_values =
      general_utils::inverse_sigmoid(target_opacity);

  opacity_.index_put_({gaussian_mask},
                      new_opacity_values.index({gaussian_mask}));

  std::cout << "[Opacity Reset] Opacity reset complete - max="
            << torch::sigmoid(opacity_).max().item<float>()
            << ", min=" << torch::sigmoid(opacity_).min().item<float>()
            << std::endl;
}

void GaussianModel::resetPositionLRAndOptimizerState(
    const torch::Tensor& gaussian_mask) {
  torch::NoGradGuard no_grad;

  if (!optimizer_) {
    std::cerr << "ERROR: Optimizer is null in "
                 "resetPositionLRAndOptimizerState!"
              << std::endl;
    return;
  }

  int num_reset = torch::sum(gaussian_mask).item<int>();
  std::cout << "[Optimizer Reset] Resetting position LR and Adam states for "
            << num_reset << " gaussians" << std::endl;

  // Reset position learning rates back to initial value
  position_lrs_.index_put_({gaussian_mask}, position_lr_init_);

  // Reset Adam optimizer states for positions (group 0 = xyz)
  auto& param_group = optimizer_->param_groups()[0];
  if (param_group.params().empty()) {
    std::cerr << "ERROR: No parameters in position group!" << std::endl;
    return;
  }

  auto& xyz_param = param_group.params()[0];
  auto& state = optimizer_->state();
  auto key = xyz_param.unsafeGetTensorImpl();

  if (state.find(key) == state.end()) {
    std::cerr << "WARNING: No optimizer state found for positions" << std::endl;
    return;
  }

  auto& param_state = static_cast<torch::optim::AdamParamState&>(*state[key]);
  torch::Tensor exp_avg = param_state.exp_avg();
  torch::Tensor exp_avg_sq = param_state.exp_avg_sq();

  // Expand mask to match xyz dimensions [N, 3]
  torch::Tensor xyz_mask = gaussian_mask.unsqueeze(1).expand({-1, 3});
  exp_avg.index_put_({xyz_mask}, 0.0f);
  exp_avg_sq.index_put_({xyz_mask}, 0.0f);

  std::cout << "[Optimizer Reset] Position LRs reset - max="
            << position_lrs_.max().item<float>()
            << ", min=" << position_lrs_.min().item<float>()
            << ", mean=" << position_lrs_.mean().item<float>() << std::endl;
}

void GaussianModel::prunePoints(torch::Tensor& mask) {
  prunePointsBase(mask);

  // Prune DiskChunGS tracking tensors
  auto valid_points_mask = ~mask;
  exist_since_iter_ = exist_since_iter_.index({valid_points_mask});
  position_lrs_ = position_lrs_.index({valid_points_mask});
  gaussian_chunk_ids_ = gaussian_chunk_ids_.index({valid_points_mask});
  gaussian_ids_ = gaussian_ids_.index({valid_points_mask});
}

void GaussianModel::densificationPostfix(
    torch::Tensor& new_xyz,
    torch::Tensor& new_features_dc,
    torch::Tensor& new_features_rest,
    torch::Tensor& new_opacities,
    torch::Tensor& new_scaling,
    torch::Tensor& new_rotation,
    torch::Tensor& new_exist_since_iter,
    torch::Tensor& new_chunk_ids,
    torch::Tensor& new_position_lrs,
    torch::Tensor& new_gaussian_ids,
    const std::vector<torch::Tensor>& loaded_exp_avg,
    const std::vector<torch::Tensor>& loaded_exp_avg_sq,
    const std::vector<int64_t>& loaded_step_counts) {
  std::vector<torch::Tensor> extension_tensors = {
      new_xyz,       new_features_dc, new_features_rest,
      new_opacities, new_scaling,     new_rotation};
  densificationPostfixBase(extension_tensors, loaded_exp_avg, loaded_exp_avg_sq,
                           loaded_step_counts);

  // Append DiskChunGS tracking tensors
  exist_since_iter_ = torch::cat({exist_since_iter_, new_exist_since_iter}, 0);
  position_lrs_ = torch::cat({position_lrs_, new_position_lrs}, 0);
  gaussian_chunk_ids_ = torch::cat({gaussian_chunk_ids_, new_chunk_ids}, 0);
  gaussian_ids_ = torch::cat({gaussian_ids_, new_gaussian_ids}, 0);
}

void GaussianModel::pruneLowOpacityGaussians(
    std::shared_ptr<GaussianKeyframe> pkf,
    const torch::Tensor& visible_gaussian_mask) {
  torch::NoGradGuard no_grad;

  torch::Tensor visible_indices = torch::where(visible_gaussian_mask)[0];

  // Keyframe camera parameters
  torch::Tensor keyframe_center = pkf->getCenter();
  float focal_length = pkf->intr_[0];  // fx
  int image_width = pkf->image_width_;

  // Gaussian parameters for visible subset
  torch::Tensor positions = getXYZ().index({visible_indices});
  torch::Tensor opacities = getOpacityActivation().index({visible_indices});
  torch::Tensor scalings = getScalingActivation().index({visible_indices});

  int n_gaussians = positions.size(0);
  torch::Tensor valid_mask = torch::ones(
      n_gaussians,
      torch::TensorOptions().dtype(torch::kBool).device(positions.device()));

  // Remove Gaussians with low opacity
  valid_mask &= opacities.squeeze(1) > 0.05f;

  // Remove Gaussians that appear too large on screen
  torch::Tensor distances =
      torch::norm(positions - keyframe_center.unsqueeze(0), 2, /*dim=*/1);
  torch::Tensor max_scaling = std::get<0>(torch::max(scalings, /*dim=*/1));
  torch::Tensor screen_size = focal_length * max_scaling / distances;
  float max_screen_size = 0.5f * static_cast<float>(image_width);
  valid_mask &= screen_size < max_screen_size;

  // Build full-model prune mask from the visible subset
  torch::Tensor full_model_prune_mask = torch::zeros(
      {getXYZ().size(0)},
      torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
  full_model_prune_mask.index_put_({visible_indices}, ~valid_mask);

  prunePoints(full_model_prune_mask);
}

void GaussianModel::deleteSparseChunks(int min_gaussians_per_chunk) {
  torch::NoGradGuard no_grad;

  if (!is_initialized_ || xyz_.size(0) == 0) {
    return;
  }

  // Count gaussians per spatial chunk
  auto [unique_chunks, inverse_indices, counts] =
      torch::_unique2(gaussian_chunk_ids_, /*sorted=*/false,
                      /*return_inverse=*/true, /*return_counts=*/true);

  torch::Tensor sparse_mask = counts < min_gaussians_per_chunk;
  torch::Tensor sparse_chunk_ids = unique_chunks.index({sparse_mask});

  if (sparse_chunk_ids.size(0) == 0) {
    return;
  }

  // Create removal mask and update tracking state
  torch::Tensor remove_mask =
      torch::isin(gaussian_chunk_ids_, sparse_chunk_ids);

  torch::Tensor keep_loaded_mask =
      ~torch::isin(chunks_loaded_from_disk_, sparse_chunk_ids);
  chunks_loaded_from_disk_ = chunks_loaded_from_disk_.index({keep_loaded_mask});

  torch::Tensor keep_disk_mask =
      ~torch::isin(chunks_on_disk_, sparse_chunk_ids);
  chunks_on_disk_ = chunks_on_disk_.index({keep_disk_mask});
  chunk_gaussian_counts_ = chunk_gaussian_counts_.index({keep_disk_mask});

  // Clear access times for deleted chunks
  auto sparse_ids_cpu = sparse_chunk_ids.cpu();
  auto sparse_ids_accessor = sparse_ids_cpu.accessor<int64_t, 1>();
  for (int64_t i = 0; i < sparse_ids_cpu.size(0); i++) {
    chunk_access_times_.erase(sparse_ids_accessor[i]);
  }

  deleteSparseChunkFiles(sparse_chunk_ids);
  prunePoints(remove_mask);
}

void GaussianModel::deleteSparseChunkFiles(const torch::Tensor& chunk_ids) {
  auto chunks_cpu = chunk_ids.cpu();
  auto accessor = chunks_cpu.accessor<int64_t, 1>();

  for (int i = 0; i < chunks_cpu.size(0); ++i) {
    int64_t chunk_id = accessor[i];
    std::string chunk_filename = getChunkFilename(decodeChunkCoord(chunk_id));

    if (std::filesystem::exists(chunk_filename)) {
      try {
        std::filesystem::remove(chunk_filename);
      } catch (const std::exception& e) {
        std::cerr << "[File Deletion] Failed to delete " << chunk_filename
                  << ": " << e.what() << std::endl;
      }
    }
  }
}
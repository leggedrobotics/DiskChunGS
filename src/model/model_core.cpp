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

GaussianModel::GaussianModel(const GaussianModelParams& model_params,
                             std::string storage_base_path,
                             float chunk_size)
    : storage_base_path_(storage_base_path),
      chunk_size_(chunk_size),
      max_gaussians_in_memory_(model_params.max_gaussians_in_memory_),
      gaussian_visibility_cache_(chunk_size) {
  // Initialize base class members
  sh_degree_ = model_params.sh_degree_;
  spatial_lr_scale_ = 1.0;
  position_lr_init_ = 0.00005;
  position_lr_decay_ = 0.99998;
  local_iteration_ = 0;

  // Device
  if (model_params.data_device_ == "cuda")
    this->device_type_ = torch::kCUDA;
  else
    this->device_type_ = torch::kCPU;

  GAUSSIAN_MODEL_INIT_TENSORS(this->device_type_)

  chunks_on_disk_ = torch::empty(
      {0}, torch::TensorOptions().dtype(torch::kInt64).device(device_type_));

  chunks_loaded_from_disk_ = torch::empty(
      {0}, torch::TensorOptions().dtype(torch::kInt64).device(device_type_));

  chunk_gaussian_counts_ = torch::empty(
      {0}, torch::TensorOptions().dtype(torch::kInt64).device(device_type_));
  gaussian_ids_ = torch::empty(
      0, torch::TensorOptions().dtype(torch::kInt64).device(device_type_));

  std::cout << "[GaussianModel] Initialized with storage path: "
            << storage_base_path_ << " and chunk size: " << chunk_size_
            << std::endl;
}

void GaussianModel::applyScaledTransformation(const float s,
                                              const Sophus::SE3f T) {
  torch::NoGradGuard no_grad;
  this->xyz_ *= s;
  torch::Tensor T_tensor =
      tensor_utils::EigenMatrix2TorchTensor(T.matrix(), device_type_)
          .transpose(0, 1);
  transformPoints(this->xyz_, T_tensor);

  this->scaling_ *= s;
  scaledTransformationPostfix(this->xyz_, this->scaling_);
}

void GaussianModel::scaledTransformationPostfix(torch::Tensor& new_xyz,
                                                torch::Tensor& new_scaling) {
  torch::Tensor optimizable_xyz = this->replaceTensorToOptimizer(new_xyz, 0);
  torch::Tensor optimizable_scaling =
      this->replaceTensorToOptimizer(new_scaling, 4);

  this->xyz_ = optimizable_xyz;
  this->scaling_ = optimizable_scaling;

  this->Tensor_vec_xyz_ = {this->xyz_};
  this->Tensor_vec_scaling_ = {this->scaling_};
}

void GaussianModel::scaledTransformVisiblePointsOfKeyframe(
    torch::Tensor& point_transformed_flags,
    const torch::Tensor& diff_pose,
    torch::Tensor& kf_world_view_transform,
    torch::Tensor& kf_full_proj_transform,
    const int kf_creation_iter,
    const int stable_num_iter_existence,
    int& num_transformed,
    const float scale) {
  torch::NoGradGuard no_grad;

  torch::Tensor points = this->getXYZ();
  torch::Tensor rots = this->getRotationActivation();

  torch::Tensor point_unstable_flags =
      torch::where(torch::abs(this->exist_since_iter_ - kf_creation_iter) <
                       stable_num_iter_existence,
                   true, false);

  scaleAndTransformThenMarkVisiblePoints(
      points, rots, point_transformed_flags, point_unstable_flags, diff_pose,
      kf_world_view_transform, kf_full_proj_transform, num_transformed, scale);

  // Postfix
  // ==================================
  // param_groups[0] = xyz_
  // param_groups[1] = feature_dc_
  // param_groups[2] = feature_rest_
  // param_groups[3] = opacity_
  // param_groups[4] = scaling_
  // param_groups[5] = rotation_
  // ==================================

  if (num_transformed > 0) {
    torch::Tensor optimizable_xyz = this->replaceTensorToOptimizer(points, 0);
    torch::Tensor optimizable_rots = this->replaceTensorToOptimizer(rots, 5);

    this->xyz_ = optimizable_xyz;
    this->rotation_ = optimizable_rots;

    this->Tensor_vec_xyz_ = {this->xyz_};
    this->Tensor_vec_rotation_ = {this->rotation_};
  }
}

void GaussianModel::addPoints(const torch::Tensor& new_xyz,
                              const torch::Tensor& new_colors,
                              const torch::Tensor& new_scales,
                              const torch::Tensor& new_opacities,
                              int iteration,
                              float spatial_lr_scale) {
  torch::NoGradGuard no_grad;

  auto [filtered_xyz, filtered_colors, filtered_scales, filtered_opacities] =
      filterPointsByChunkDensity(new_xyz, new_colors, new_scales, new_opacities,
                                 new_gaussian_chunk_density_);

  if (filtered_xyz.size(0) == 0) {
    std::cout
        << "[Gaussian Model] Warning: No points passed chunk density filter"
        << std::endl;
    return;
  }

  // Only care about existing disk chunks that need loading
  torch::Tensor affected_chunks =
      std::get<0>(torch::_unique2(computeChunkIds(filtered_xyz, chunk_size_)));
  torch::Tensor existing_disk_chunks =
      torch::isin(affected_chunks, chunks_on_disk_);
  torch::Tensor unloaded_disk_chunks = affected_chunks.index(
      {existing_disk_chunks &
       ~torch::isin(affected_chunks, chunks_loaded_from_disk_)});

  if (unloaded_disk_chunks.size(0) > 0) {
    // Load existing disk chunks that we're about to add points to
    loadChunks(unloaded_disk_chunks);
  }

  if (!is_initialized_) {
    // First call - initialize the model
    initializeFromPoints(filtered_xyz, filtered_colors, filtered_scales,
                         filtered_opacities, iteration);
  } else {
    // Subsequent calls - append to existing model
    appendPoints(filtered_xyz, filtered_colors, filtered_scales,
                 filtered_opacities, iteration);
  }
}

void GaussianModel::initializeFromPoints(const torch::Tensor& initial_xyz,
                                         const torch::Tensor& initial_colors,
                                         const torch::Tensor& initial_scales,
                                         const torch::Tensor& initial_opacities,
                                         int iteration) {
  torch::NoGradGuard no_grad;
  std::cout << "[Gaussian Model] Initializing from points: "
            << initial_xyz.sizes() << std::endl;

  torch::Tensor fused_color = sh_utils::RGB2SH(initial_colors);
  auto temp = this->sh_degree_ + 1;
  torch::Tensor features = torch::zeros(
      {fused_color.size(0), 3, temp * temp},
      torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
  features.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3), 0}) =
      fused_color;
  features.index({torch::indexing::Slice(),
                  torch::indexing::Slice(3, features.size(1)),
                  torch::indexing::Slice(1, features.size(2))}) = 0.0f;

  torch::Tensor scales = initial_scales;

  torch::Tensor rots = torch::zeros(
      {initial_xyz.size(0), 4}, torch::TensorOptions().device(device_type_));
  rots.index({torch::indexing::Slice(), 0}) = 1;

  torch::Tensor opacities = initial_opacities;

  this->exist_since_iter_ = torch::full(
      {initial_xyz.size(0)}, iteration,
      torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

  this->xyz_ = initial_xyz.requires_grad_();
  this->features_dc_ =
      features
          .index({torch::indexing::Slice(), torch::indexing::Slice(),
                  torch::indexing::Slice(0, 1)})
          .transpose(1, 2)
          .contiguous()
          .requires_grad_();
  this->features_rest_ =
      features
          .index({torch::indexing::Slice(), torch::indexing::Slice(),
                  torch::indexing::Slice(1, features.size(2))})
          .transpose(1, 2)
          .contiguous()
          .requires_grad_();
  this->scaling_ = scales.requires_grad_();
  this->rotation_ = rots.requires_grad_();
  this->opacity_ = opacities.requires_grad_();

  gaussian_chunk_ids_ = computeChunkIds(initial_xyz, chunk_size_);

  gaussian_ids_ = torch::arange(
      next_gaussian_id_, next_gaussian_id_ + initial_xyz.size(0),
      torch::TensorOptions().dtype(torch::kInt64).device(device_type_));
  next_gaussian_id_ += initial_xyz.size(0);

  GAUSSIAN_MODEL_TENSORS_TO_VEC

  is_initialized_ = true;
}

void GaussianModel::appendPoints(const torch::Tensor& new_xyzs,
                                 const torch::Tensor& new_colors,
                                 const torch::Tensor& new_scales,
                                 const torch::Tensor& new_opacities,
                                 int iteration) {
  torch::NoGradGuard no_grad;
  auto num_new_points = new_xyzs.size(0);
  if (num_new_points == 0) return;

  torch::Tensor new_fused_colors = sh_utils::RGB2SH(new_colors);
  auto temp = this->sh_degree_ + 1;
  torch::Tensor features = torch::zeros(
      {new_fused_colors.size(0), 3, temp * temp},
      torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
  features.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3), 0}) =
      new_fused_colors;
  features.index({torch::indexing::Slice(),
                  torch::indexing::Slice(3, features.size(1)),
                  torch::indexing::Slice(1, features.size(2))}) = 0.0f;

  torch::Tensor scales = new_scales;

  torch::Tensor rots = torch::zeros(
      {new_xyzs.size(0), 4}, torch::TensorOptions().device(device_type_));
  rots.index({torch::indexing::Slice(), 0}) = 1;

  torch::Tensor new_exist_since_iter = torch::full(
      {new_xyzs.size(0)}, iteration,
      torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

  auto new_features_dc =
      features
          .index({torch::indexing::Slice(), torch::indexing::Slice(),
                  torch::indexing::Slice(0, 1)})
          .transpose(1, 2)
          .contiguous();
  auto new_features_rest =
      features
          .index({torch::indexing::Slice(), torch::indexing::Slice(),
                  torch::indexing::Slice(1, features.size(2))})
          .transpose(1, 2)
          .contiguous();
  auto new_opacities_tensor = new_opacities;
  auto new_xyz_tensor = new_xyzs;
  auto new_scaling = scales;
  auto new_rotation = rots;

  torch::Tensor new_position_lrs =
      torch::full({new_xyzs.size(0)}, position_lr_init_,
                  torch::TensorOptions().device(device_type_));

  torch::Tensor new_gaussian_ids = torch::arange(
      next_gaussian_id_, next_gaussian_id_ + new_xyzs.size(0),
      torch::TensorOptions().dtype(torch::kInt64).device(device_type_));
  next_gaussian_id_ += new_xyzs.size(0);

  torch::Tensor new_chunk_ids = computeChunkIds(new_xyzs, chunk_size_);

  densificationPostfix(new_xyz_tensor, new_features_dc, new_features_rest,
                       new_opacities_tensor, new_scaling, new_rotation,
                       new_exist_since_iter, new_chunk_ids, new_position_lrs,
                       new_gaussian_ids);
}

std::vector<ChunkCoord> GaussianModel::frustumCullChunks(
    std::shared_ptr<GaussianKeyframe> keyframe,
    bool use_cache) {
  // Delegate to standalone function with our cache
  FrustumCullingCache* cache_ptr =
      use_cache ? &gaussian_visibility_cache_ : nullptr;
  return ::frustumCullChunks(keyframe, chunk_size_, cache_ptr);
}

torch::Tensor GaussianModel::cullVisibleGaussians(
    std::shared_ptr<GaussianKeyframe> keyframe,
    bool manage_memory) {
  torch::NoGradGuard no_grad;

  // Frustum cull chunks
  std::vector<ChunkCoord> visible_chunks =
      frustumCullChunks(keyframe, /*use_cache=*/true);

  if (visible_chunks.empty()) {
    // Return all-false mask
    return torch::zeros(
        {xyz_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(device_type_));
  }

  torch::Tensor visible_chunk_coords =
      chunkCoordVectorToTensor(visible_chunks, device_type_);
  torch::Tensor visible_chunk_ids =
      encodeChunkCoordsTensor(visible_chunk_coords);

  if (manage_memory && visible_chunk_ids.size(0) > 0) {
    loadChunks(visible_chunk_ids);
  }

  // Convert chunk visibility to gaussian visibility
  torch::Tensor chunk_visibility_mask =
      createGaussianMaskFromChunks(visible_chunk_ids);

  updateChunkAccess(visible_chunk_ids);

  return chunk_visibility_mask;
}

torch::Tensor GaussianModel::createGaussianMaskFromChunks(
    const torch::Tensor& visible_chunk_ids) {
  if (visible_chunk_ids.size(0) == 0) {
    return torch::zeros(
        {gaussian_chunk_ids_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(device_type_));
  }

  return torch::isin(gaussian_chunk_ids_, visible_chunk_ids);
}

void GaussianModel::initializeEmpty(float spatial_lr_scale) {
  std::cout << "[Gaussian Model] Initializing empty model for loading"
            << std::endl;

  this->spatial_lr_scale_ = spatial_lr_scale;

  // Initialize with empty tensors but correct shapes
  this->xyz_ =
      torch::empty(
          {0, 3},
          torch::TensorOptions().dtype(torch::kFloat).device(device_type_))
          .requires_grad_();
  this->features_dc_ =
      torch::empty(
          {0, 1, 3},
          torch::TensorOptions().dtype(torch::kFloat).device(device_type_))
          .requires_grad_();
  this->features_rest_ =
      torch::empty(
          {0, (sh_degree_ + 1) * (sh_degree_ + 1) - 1, 3},
          torch::TensorOptions().dtype(torch::kFloat).device(device_type_))
          .requires_grad_();
  this->scaling_ =
      torch::empty(
          {0, 3},
          torch::TensorOptions().dtype(torch::kFloat).device(device_type_))
          .requires_grad_();
  this->rotation_ =
      torch::empty(
          {0, 4},
          torch::TensorOptions().dtype(torch::kFloat).device(device_type_))
          .requires_grad_();
  this->opacity_ =
      torch::empty(
          {0, 1},
          torch::TensorOptions().dtype(torch::kFloat).device(device_type_))
          .requires_grad_();

  // Initialize auxiliary tensors
  this->exist_since_iter_ = torch::empty(
      {0}, torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
  this->gaussian_chunk_ids_ = torch::empty(
      {0}, torch::TensorOptions().dtype(torch::kInt64).device(device_type_));

  // Initialize tensor vectors for optimizer
  GAUSSIAN_MODEL_TENSORS_TO_VEC

  // Mark as initialized
  is_initialized_ = true;

  std::cout << "[Gaussian Model] Empty model initialized" << std::endl;
}
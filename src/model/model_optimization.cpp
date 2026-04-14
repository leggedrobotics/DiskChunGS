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

void GaussianModel::trainingSetup(
    const GaussianOptimizationParams& training_args) {
  position_lr_init_ = training_args.position_lr_init_ * spatial_lr_scale_;
  position_lr_decay_ = training_args.position_lr_decay_;
  position_lr_min_ = position_lr_init_ * 0.1f * spatial_lr_scale_;

  torch::optim::AdamOptions adam_options;
  adam_options.set_lr(0.0);
  adam_options.eps() = 1e-15;

  optimizer_.reset(new SparseGaussianAdam(Tensor_vec_xyz_, adam_options));
  optimizer_->param_groups()[0].options().set_lr(0.0f);

  // Per-Gaussian position learning rates
  int num_gaussians = getXYZ().size(0);
  position_lrs_ = torch::full(
      {num_gaussians}, position_lr_init_,
      torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

  // Remaining parameter groups use scalar learning rates
  optimizer_->add_param_group(Tensor_vec_feature_dc_);
  optimizer_->param_groups()[1].options().set_lr(training_args.feature_lr_);

  optimizer_->add_param_group(Tensor_vec_feature_rest_);
  optimizer_->param_groups()[2].options().set_lr(training_args.feature_lr_ /
                                                 20.0);

  optimizer_->add_param_group(Tensor_vec_opacity_);
  optimizer_->param_groups()[3].options().set_lr(training_args.opacity_lr_);

  optimizer_->add_param_group(Tensor_vec_scaling_);
  optimizer_->param_groups()[4].options().set_lr(training_args.scaling_lr_);

  optimizer_->add_param_group(Tensor_vec_rotation_);
  optimizer_->param_groups()[5].options().set_lr(training_args.rotation_lr_);
}

void GaussianModel::updateLearningRates(const torch::Tensor& visibility) {
  if (visibility.size(0) != position_lrs_.size(0)) {
    throw std::runtime_error(
        "Visibility tensor size doesn't match position_lrs_ size");
  }

  position_lrs_.index_put_(
      {visibility}, position_lrs_.index({visibility}) * position_lr_decay_);
  position_lrs_.clamp_min_(position_lr_min_);
}

void GaussianModel::optimizerStep(torch::Tensor& visibility,
                                   const uint32_t N) {
  torch::NoGradGuard no_grad;

  auto& param_groups = optimizer_->param_groups();

  for (size_t group_idx = 0; group_idx < param_groups.size(); ++group_idx) {
    auto& group = param_groups[group_idx];
    auto& param = group.params()[0];

    if (!param.grad().defined()) continue;

    // Lazily initialize Adam state
    auto& state = optimizer_->state();
    auto key = param.unsafeGetTensorImpl();
    if (state.find(key) == state.end()) {
      auto new_state = std::make_unique<torch::optim::AdamParamState>();
      new_state->step(0);
      new_state->exp_avg(torch::zeros_like(param));
      new_state->exp_avg_sq(torch::zeros_like(param));
      state[key] = std::move(new_state);
    }

    auto& param_state = static_cast<torch::optim::AdamParamState&>(*state[key]);
    auto options = static_cast<torch::optim::AdamOptions&>(group.options());

    // Group 0 uses per-Gaussian position LRs; others use a scalar LR
    torch::Tensor lr_tensor;
    if (group_idx == 0) {
      lr_tensor = position_lrs_;
    } else {
      lr_tensor = torch::tensor(
          {static_cast<float>(group.options().get_lr())},
          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    }

    const uint32_t M = param.numel() / N;
    adamUpdate(param, param.grad(), param_state.exp_avg(),
               param_state.exp_avg_sq(), visibility, lr_tensor,
               std::get<0>(options.betas()), std::get<1>(options.betas()),
               options.eps(), N, M);
  }

  updateLearningRates(visibility);
}

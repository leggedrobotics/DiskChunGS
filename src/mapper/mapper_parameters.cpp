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

#include "gaussian_mapper.h"

int GaussianMapper::getIteration() {
  std::unique_lock<std::mutex> lock(mutex_status_);
  return iteration_;
}

void GaussianMapper::increaseIteration(const int inc) {
  std::unique_lock<std::mutex> lock(mutex_status_);
  iteration_ += inc;
}

float GaussianMapper::positionLearningRateInit() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.position_lr_init_;
}

float GaussianMapper::featureLearningRate() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.feature_lr_;
}

float GaussianMapper::opacityLearningRate() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.opacity_lr_;
}

float GaussianMapper::scalingLearningRate() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.scaling_lr_;
}

float GaussianMapper::rotationLearningRate() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.rotation_lr_;
}

float GaussianMapper::lambdaDssim() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.lambda_dssim_;
}

float GaussianMapper::lambdaDepth() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.lambda_depth_;
}

int GaussianMapper::newKeyframeTimesOfUse() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return new_keyframe_times_of_use_;
}

int GaussianMapper::stableNumIterExistence() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return stable_num_iter_existence_;
}

bool GaussianMapper::isKeepingTraining() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return keep_training_;
}

void GaussianMapper::setLambdaDssim(const float lambda_dssim) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  opt_params_.lambda_dssim_ = lambda_dssim;
}

void GaussianMapper::setNewKeyframeTimesOfUse(const int times) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  new_keyframe_times_of_use_ = times;
}

void GaussianMapper::setStableNumIterExistence(const int niter) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  stable_num_iter_existence_ = niter;
}

void GaussianMapper::setKeepTraining(const bool keep) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  keep_training_ = keep;
}

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

#include <filesystem>
#include <iostream>

#include "gaussian_mapper.h"

// Helper template to read config values with less boilerplate
template <typename T>
T readConfig(const cv::FileStorage& fs, const std::string& key) {
  return fs[key].operator T();
}

// Helper for reading boolean values (converts int to bool)
bool readConfigBool(const cv::FileStorage& fs, const std::string& key) {
  return readConfig<int>(fs, key) != 0;
}

void copyFolder(const std::filesystem::path& source,
                const std::filesystem::path& destination) {
  // Create the destination directory if it doesn't exist
  std::filesystem::create_directories(destination);

  // Iterate through the source directory
  for (const auto& entry : std::filesystem::directory_iterator(source)) {
    const auto& path = entry.path();
    const auto destPath = destination / path.filename();

    if (std::filesystem::is_directory(path)) {
      // Recursively copy subdirectories
      copyFolder(path, destPath);
    } else {
      // Copy files
      std::filesystem::copy_file(
          path, destPath, std::filesystem::copy_options::overwrite_existing);
    }
  }
}

void GaussianMapper::readConfigFromFile(std::filesystem::path cfg_path) {
  cv::FileStorage settings_file(cfg_path.string().c_str(),
                                cv::FileStorage::READ);
  if (!settings_file.isOpened()) {
    std::cerr << "[Gaussian Mapper]Failed to open settings file at: "
              << cfg_path << std::endl;
    exit(-1);
  }

  std::cout << "[Gaussian Mapper]Reading parameters from " << cfg_path
            << std::endl;
  std::unique_lock<std::mutex> lock(mutex_settings_);

  // ========== Model Parameters ==========
  model_params_.sh_degree_ = readConfig<int>(settings_file, "Model.sh_degree");
  model_params_.white_background_ =
      readConfigBool(settings_file, "Model.white_background");
  model_params_.max_gaussians_in_memory_ =
      readConfig<int>(settings_file, "Model.max_gaussians_in_memory");
  init_proba_scaler_ =
      readConfig<float>(settings_file, "Model.init_proba_scaler");
  downsample_for_sampling_ =
      readConfigBool(settings_file, "Model.downsample_for_sampling");

  // ========== Camera Parameters ==========
  z_near_ = readConfig<float>(settings_file, "Camera.z_near");
  z_far_ = readConfig<float>(settings_file, "Camera.z_far");

  // ========== Mapper Parameters ==========
  min_depth_ = readConfig<float>(settings_file, "Mapper.min_depth_");
  max_depth_ = readConfig<float>(settings_file, "Mapper.max_depth_");
  min_num_initial_map_kfs_ = static_cast<unsigned long>(
      readConfig<int>(settings_file, "Mapper.min_num_initial_map_kfs"));
  new_keyframe_times_of_use_ =
      readConfig<int>(settings_file, "Mapper.new_keyframe_times_of_use");
  local_BA_increased_times_of_use_ =
      readConfig<int>(settings_file, "Mapper.local_BA_increased_times_of_use");
  loop_closure_increased_times_of_use_ = readConfig<int>(
      settings_file, "Mapper.loop_closure_increased_times_of_use_");
  large_rot_th_ =
      readConfig<float>(settings_file, "Mapper.large_rotation_threshold");
  large_trans_th_ =
      readConfig<float>(settings_file, "Mapper.large_translation_threshold");
  stable_num_iter_existence_ =
      readConfig<int>(settings_file, "Mapper.stable_num_iter_existence");

  // ========== External Mode Parameters ==========
  min_keyframe_translation_ =
      readConfig<float>(settings_file, "External.min_keyframe_translation");
  min_keyframe_rotation_ =
      readConfig<float>(settings_file, "External.min_keyframe_rotation");
  min_keyframe_time_ =
      readConfig<float>(settings_file, "External.min_keyframe_time");

  // ========== Pipeline Parameters ==========
  pipe_params_.convert_SHs_ =
      readConfigBool(settings_file, "Pipeline.convert_SHs");
  pipe_params_.compute_cov3D_ =
      readConfigBool(settings_file, "Pipeline.compute_cov3D");

  // ========== Gaussian Pyramid Parameters ==========
  num_gaus_pyramid_sub_levels_ =
      readConfig<int>(settings_file, "GausPyramid.num_levels");
  int sub_level_times_of_use =
      readConfig<int>(settings_file, "GausPyramid.level_times_of_use");
  kf_gaus_pyramid_times_of_use_.resize(num_gaus_pyramid_sub_levels_);
  kf_gaus_pyramid_factors_.resize(num_gaus_pyramid_sub_levels_);
  for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
    kf_gaus_pyramid_times_of_use_[l] = sub_level_times_of_use;
    kf_gaus_pyramid_factors_[l] = std::pow(0.5f, l);
  }

  // ========== Recording Parameters ==========
  keyframe_record_interval_ =
      readConfig<int>(settings_file, "Record.keyframe_record_interval");
  all_keyframes_record_interval_ =
      readConfig<int>(settings_file, "Record.all_keyframes_record_interval");
  record_rendered_image_ =
      readConfigBool(settings_file, "Record.record_rendered_image");
  record_ground_truth_image_ =
      readConfigBool(settings_file, "Record.record_ground_truth_image");
  record_loss_image_ =
      readConfigBool(settings_file, "Record.record_loss_image");
  training_report_interval_ =
      readConfig<int>(settings_file, "Record.training_report_interval");
  record_loop_ply_ = readConfigBool(settings_file, "Record.record_loop_ply");

  // ========== Optimization Parameters ==========
  opt_params_.iterations_ =
      readConfig<int>(settings_file, "Optimization.max_num_iterations");
  opt_params_.position_lr_init_ =
      readConfig<float>(settings_file, "Optimization.position_lr_init");
  opt_params_.position_lr_decay_ =
      readConfig<float>(settings_file, "Optimization.position_lr_decay");
  opt_params_.feature_lr_ =
      readConfig<float>(settings_file, "Optimization.feature_lr");
  opt_params_.opacity_lr_ =
      readConfig<float>(settings_file, "Optimization.opacity_lr");
  opt_params_.scaling_lr_ =
      readConfig<float>(settings_file, "Optimization.scaling_lr");
  opt_params_.rotation_lr_ =
      readConfig<float>(settings_file, "Optimization.rotation_lr");
  opt_params_.pose_lr_ =
      readConfig<float>(settings_file, "Optimization.pose_lr");
  opt_params_.exposure_lr_ =
      readConfig<float>(settings_file, "Optimization.exposure_lr");
  opt_params_.depth_scale_bias_lr_ =
      readConfig<float>(settings_file, "Optimization.depth_scale_bias_lr");
  opt_params_.smooth_l1_ =
      readConfigBool(settings_file, "Optimization.smooth_l1");

  opt_params_.lambda_dssim_ =
      readConfig<float>(settings_file, "Optimization.lambda_dssim");
  opt_params_.lambda_depth_ =
      readConfig<float>(settings_file, "Optimization.lambda_depth");
  opt_params_.auto_distribute_ =
      readConfig<int>(settings_file, "Optimization.auto_distribute");
  exposure_optimization_ =
      readConfig<int>(settings_file, "Optimization.exposure_optimization");

  // ========== Viewer Parameters ==========
  rendered_image_viewer_scale_ =
      readConfig<float>(settings_file, "GaussianViewer.image_scale");
  rendered_image_viewer_scale_main_ =
      readConfig<float>(settings_file, "GaussianViewer.image_scale_main");

  // ========== Chunking Parameters ==========
  chunk_size_ = readConfig<float>(settings_file, "Chunking.chunk_size");
  keyframe_selection_chunk_size_ = readConfig<float>(
      settings_file, "Chunking.keyframe_selection_chunk_size");
}

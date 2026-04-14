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

#include <torch/torch.h>

#include <filesystem>
#include <iostream>

#include "gaussian_mapper.h"
#include "gaussian_splatting/depth/depth_alignment.h"
#include "geometry/operate_points.h"
#include "utils/tensor_utils.h"

torch::Tensor GaussianMapper::computeLoGProbability(
    const torch::Tensor& image) {
  return depth_alignment::computeLoGProbability(image, disc_kernel_);
}

void GaussianMapper::initializeLaplacianOfGaussianKernel() {
  disc_kernel_ = depth_alignment::createDiscKernel(LOG_KERNEL_RADIUS,
                                                   device_type_);
}

void GaussianMapper::initializeStereoDepthEstimator() {
  // Construct model path for configured resolution
  std::string onnx_path =
      std::string(DEPTH_MODEL_BASE_DIR) +
      "fast_acvnet_plus_kitti_2015_opset16_" +
      std::to_string(STEREO_MODEL_HEIGHT) + "x" +
      std::to_string(STEREO_MODEL_WIDTH) + ".onnx";

  this->stereo_depth_estimator_ = std::make_shared<StereoDepth>(onnx_path);
}

void GaussianMapper::initializeMonocularDepthEstimator() {
  std::string onnx_path =
      std::string(DEPTH_MODEL_BASE_DIR) + "depth_anything_v2_vitl.onnx";

  this->monocular_depth_estimator_ = std::make_shared<MonoDepth>(onnx_path);
}

torch::Tensor GaussianMapper::sampleConf(const torch::Tensor& mono_depth_conf,
                                         const torch::Tensor& uv,
                                         int width,
                                         int height) {
  return depth_alignment::sampleConf(mono_depth_conf, uv, width, height);
}

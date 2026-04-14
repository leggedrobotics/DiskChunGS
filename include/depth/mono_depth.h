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
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <tuple>
#include <vector>

#include "gaussian_splatting/depth/depth_alignment.h"
#include "third_party/depth-anything-tensorrt/depth_anything.h"
#include "utils/tensor_utils.h"

/**
 * @brief MonoDepth estimation class using TensorRT DepthAnything
 */
class MonoDepth {
 public:
  MonoDepth(const std::string& model_path);
  ~MonoDepth() = default;

  /**
   * @brief Estimate depth from monocular image
   * @return Tuple of depth tensor and confidence tensor
   */
  std::tuple<torch::Tensor, torch::Tensor> estimate_depth(
      const cv::Mat& image,
      float focal_length = 0.0f);

  /**
   * @brief Align depth using sparse keypoints.
   */
  torch::Tensor align_depth(const torch::Tensor& mono_depth_map,
                            const std::vector<float>& keypoint_pixels,
                            const std::vector<float>& keypoint_depths,
                            int width,
                            int height) const;

 private:
  std::unique_ptr<DepthAnything> depth_anything_;

  int input_height_;
  int input_width_;
  int img_height_;
  int img_width_;

  static constexpr int DEFAULT_INPUT_HEIGHT = 518;
  static constexpr int DEFAULT_INPUT_WIDTH = 518;

  void initialize_model(const std::string& model_path);
};

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

#include "depth/mono_depth.h"

#include <NvInfer.h>

#include "utils/depth_utils.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cout << "[TensorRT] " << msg << std::endl;
    }
  }
};
}  // anonymous namespace

MonoDepth::MonoDepth(const std::string& model_path) {
  initialize_model(model_path);
}

void MonoDepth::initialize_model(const std::string& user_model_path) {
  std::string base_filepath = user_model_path;
  size_t pos = base_filepath.rfind(".onnx");
  if (pos != std::string::npos) {
    base_filepath = base_filepath.substr(0, pos);
  }

  std::string engine_path = base_filepath + ".engine";

  std::string model_path;
  if (std::filesystem::exists(engine_path)) {
    std::cout << "Cached depth estimation engine file found." << std::endl;
    model_path = engine_path;
  } else {
    std::cout
        << "No engine file found (normal for first time run). Building engine "
           "from ONNX. This can take a while (especially on Jetson)"
        << std::endl;
    model_path = user_model_path;
  }

  std::ifstream file(model_path);
  if (!file.good()) {
    throw std::runtime_error("Model file not found: " + model_path);
  }
  file.close();

  std::cout << "Loading TensorRT model: " << model_path << std::endl;

  try {
    depth_anything_ = std::make_unique<DepthAnything>();
    static Logger logger;
    depth_anything_->init(model_path, logger);
    std::cout << "TensorRT DepthAnything model initialized successfully"
              << std::endl;
    input_height_ = DEFAULT_INPUT_HEIGHT;
    input_width_ = DEFAULT_INPUT_WIDTH;
  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to initialize TensorRT model: " +
                             std::string(e.what()));
  }
}

std::tuple<torch::Tensor, torch::Tensor> MonoDepth::estimate_depth(
    const cv::Mat& image,
    float focal_length) {
  if (!depth_anything_) {
    throw std::runtime_error("DepthAnything model not initialized");
  }

  img_height_ = image.rows;
  img_width_ = image.cols;

  cv::Mat input_image;
  if (image.type() != CV_8UC3) {
    if (image.type() == CV_32FC3) {
      image.convertTo(input_image, CV_8UC3, 255.0);
    } else {
      image.convertTo(input_image, CV_8UC3);
    }
  } else {
    input_image = image.clone();
  }

  cv::Mat raw_depth = depth_anything_->predict(input_image);

  torch::Tensor depth =
      tensor_utils::cvMat2TorchTensor_Float32(raw_depth, torch::kCUDA);

  auto [t, s] = depth_alignment::get_t_s(depth);
  depth = (depth - t) / s;

  if (depth.dim() == 2) {
    depth = depth.unsqueeze(0).unsqueeze(0);
  } else if (depth.dim() == 3) {
    depth = depth.unsqueeze(0);
  }

  torch::Tensor confidence = depth_utils::computeDepthConfidence(depth);

  return std::make_tuple(depth, confidence);
}

torch::Tensor MonoDepth::align_depth(
    const torch::Tensor& mono_depth_map,
    const std::vector<float>& keypoint_pixels,
    const std::vector<float>& keypoint_depths,
    int width,
    int height) const {
  return depth_alignment::align_depth(mono_depth_map, keypoint_pixels,
                                      keypoint_depths, width, height);
}

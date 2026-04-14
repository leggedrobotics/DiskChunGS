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

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <fstream>
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

/**
 * @brief Fast ACVNet depth estimation class using TensorRT
 *
 * This class provides stereo depth estimation using the Fast ACVNet model
 * with TensorRT for inference and OpenCV for image processing.
 */
class StereoDepth {
 public:
  /**
   * @brief Constructor
   * @param model_path Path to the ONNX model file
   * @param max_dist Maximum distance for depth visualization
   */
  StereoDepth(const std::string& model_path);

  /**
   * @brief Destructor
   */
  ~StereoDepth();

  /**
   * @brief Estimate depth from stereo images
   * @param left_img Left stereo image
   * @param right_img Right stereo image
   * @return Disparity map
   */
  cv::Mat estimate_depth(const cv::Mat& left_img, const cv::Mat& right_img);

  /**
   * @brief Estimate depth from stereo images and convert to metric depth
   * @param left_img Left stereo image
   * @param right_img Right stereo image
   * @param focal_length Camera focal length in pixels (for original image
   * resolution)
   * @param baseline Stereo baseline distance in meters
   * @return Depth map in meters
   */
  cv::Mat estimate_metric_depth(const cv::Mat& left_img,
                                const cv::Mat& right_img,
                                const float focal_length,
                                const float baseline);

 private:
  /**
   * @brief TensorRT Logger class
   */
  class Logger : public nvinfer1::ILogger {
   public:
    void log(Severity severity, const char* msg) noexcept override {
      if (severity <= Severity::kWARNING) {
        std::cout << msg << std::endl;
      }
    }
  };

  // TensorRT components
  Logger logger_;
  std::unique_ptr<nvinfer1::IBuilder> builder_;
  std::unique_ptr<nvinfer1::INetworkDefinition> network_;
  std::unique_ptr<nvinfer1::IBuilderConfig> config_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  // Model information
  int input_height_;
  int input_width_;
  int input_size_;  // Pre-calculated input size
  int img_height_;
  int img_width_;

  // GPU memory buffers
  void* buffers_[3];  // left_input, right_input, output
  cudaStream_t stream_;
  float* output_data_;

  // Pinned host memory for faster transfers
  float* pinned_left_input_;
  float* pinned_right_input_;

  // GPU preprocessing buffers
  cv::cuda::GpuMat gpu_resized_;
  cv::cuda::GpuMat gpu_float_;
  cv::cuda::GpuMat gpu_normalized_;
  std::vector<cv::cuda::GpuMat> gpu_channels_;

  // Results
  cv::Mat disparity_map_;
  cv::Mat depth_map_;

  // Configuration
  float max_dist_;
  std::string engine_cache_path_;

  // Constants
  static constexpr const char* DEFAULT_ENGINE_CACHE_DIR =
      "/workspace/repo/models/";

  /**
   * @brief Initialize TensorRT model
   * @param model_path Path to ONNX model file
   */
  void initialize_model(const std::string& model_path);

  /**
   * @brief Build TensorRT engine from ONNX model
   * @param onnx_path Path to ONNX model file
   */
  void build_engine(const std::string& onnx_path);

  /**
   * @brief Load TensorRT engine from cache
   * @param engine_path Path to engine cache file
   */
  bool load_engine(const std::string& engine_path);

  /**
   * @brief Save TensorRT engine to cache
   * @param engine_path Path to save engine cache
   */
  bool save_engine(const std::string& engine_path);

  /**
   * @brief Allocate GPU memory buffers
   */
  void allocate_buffers();

  /**
   * @brief Free GPU memory buffers
   */
  void free_buffers();

  /**
   * @brief Get model input/output dimensions
   */
  void get_model_info();

  /**
   * @brief Preprocess input image on GPU and write to pinned memory buffer
   *
   * Performs resize, normalization, and channel reordering on GPU. Output is
   * written to pinned host memory for faster CPU-GPU transfers during
   * inference.
   *
   * @param img Input image
   * @param output_buffer Pinned memory buffer to write to
   */
  void prepare_input_optimized(const cv::Mat& img, float* output_buffer);

  /**
   * @brief Run TensorRT inference
   * @return Disparity map
   */
  cv::Mat inference_tensorrt();
};
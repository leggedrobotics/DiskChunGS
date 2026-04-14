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

#pragma once

#include <opencv2/imgproc/types_c.h>
#include <torch/torch.h>

#include <Eigen/Geometry>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

namespace tensor_utils {

inline void deleter(void* arg) {}

inline torch::Tensor cvMat2TorchTensor_Float32(const cv::Mat& mat,
                                               torch::DeviceType device_type) {
  // First make sure we have a continuous matrix
  cv::Mat continuous_mat;
  if (!mat.isContinuous()) {
    continuous_mat = mat.clone();
  } else {
    continuous_mat = mat;
  }

  // Convert to float32 if not already
  cv::Mat float_mat;
  if (continuous_mat.type() != CV_32FC1 && continuous_mat.type() != CV_32FC3) {
    continuous_mat.convertTo(
        float_mat, continuous_mat.channels() == 1 ? CV_32FC1 : CV_32FC3,
        1.0 / 255.0);  // Scale to 0-1 range if needed
  } else {
    float_mat = continuous_mat;
  }

  // Set explicit tensor options with proper dtype
  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat32)
                     .layout(torch::kStrided)
                     .device(torch::kCPU);  // Always create on CPU first

  torch::Tensor cpu_tensor;

  try {
    switch (float_mat.channels()) {
      case 1: {
        cpu_tensor =
            torch::from_blob(float_mat.data, {float_mat.rows, float_mat.cols},
                             options)
                .clone();  // Clone to own data
        break;
      }
      case 3: {
        cpu_tensor = torch::from_blob(
                         float_mat.data,
                         {float_mat.rows, float_mat.cols, float_mat.channels()},
                         options)
                         .clone();  // Clone to own data
        cpu_tensor = cpu_tensor.permute({2, 0, 1});
        break;
      }
      default:
        std::cerr << "Mat has " << float_mat.channels()
                  << " channels (unsupported)" << std::endl;
        return torch::empty({0}, options);
    }

    // Only transfer to device if needed
    if (device_type != torch::kCPU) {
      // Add error handling for CUDA transfer
      try {
        return cpu_tensor.to(device_type, true).contiguous();
      } catch (const c10::Error& e) {
        std::cerr << "CUDA transfer error: " << e.what() << std::endl;
        // Fall back to CPU
        return cpu_tensor.contiguous();
      }
    }

    return cpu_tensor.contiguous();

  } catch (const std::exception& e) {
    std::cerr << "Exception in tensor conversion: " << e.what() << std::endl;
    std::cerr << "Mat info - rows: " << float_mat.rows
              << ", cols: " << float_mat.cols
              << ", channels: " << float_mat.channels()
              << ", type: " << float_mat.type() << std::endl;
    return torch::empty({0}, options);
  }
}

/**
 * @brief Convert a PyTorch tensor to OpenCV Mat (float32)
 *
 * @param tensor Input tensor {channels, rows, cols}
 * @return cv::Mat Output matrix {rows, cols, channels}
 */
inline cv::Mat torchTensor2CvMat_Float32(const torch::Tensor& tensor) {
  cv::Mat mat;

  // Move to CPU and ensure contiguous memory layout first
  torch::Tensor mat_tensor = tensor.to(torch::kCPU).contiguous();

  switch (mat_tensor.ndimension()) {
    case 2: {
      mat = cv::Mat(mat_tensor.size(0), mat_tensor.size(1), CV_32FC1);
      std::memcpy(mat.data, mat_tensor.data_ptr<float>(),
                  mat_tensor.numel() * sizeof(float));
    } break;

    case 3: {
      mat_tensor = mat_tensor.permute({1, 2, 0}).contiguous();
      mat = cv::Mat(mat_tensor.size(0), mat_tensor.size(1), CV_32FC3);
      std::memcpy(mat.data, mat_tensor.data_ptr<float>(),
                  mat_tensor.numel() * sizeof(float));
    } break;

    default:
      std::cerr << "The tensor has unsupported number of dimensions!"
                << std::endl;
      break;
  }

  return mat;
}

/**
 * @brief
 *
 * @param cv::cuda::GpuMat {rows, cols, channels}
 * @return torch::Tensor {channels, rows, cols}
 */
/**
 * @brief Convert cv::cuda::GpuMat to torch::Tensor with enhanced robustness
 *
 * @param mat cv::cuda::GpuMat {rows, cols, channels}
 * @param device_type Target device for tensor (default: torch::kCUDA)
 * @return torch::Tensor {channels, rows, cols}
 */
inline torch::Tensor cvGpuMat2TorchTensor_Float32(
    cv::cuda::GpuMat& mat,
    torch::DeviceType device_type = torch::kCUDA) {
  // First make sure we're working with a float32 GPU mat
  cv::cuda::GpuMat float_mat;
  if (mat.type() != CV_32FC1 && mat.type() != CV_32FC3) {
    // Convert to float32 and scale to 0-1 range if needed
    cv::cuda::GpuMat tmp_mat;
    mat.convertTo(tmp_mat, mat.channels() == 1 ? CV_32FC1 : CV_32FC3,
                  1.0 / 255.0);
    float_mat = tmp_mat;
  } else {
    float_mat = mat;
  }

  torch::Tensor tensor;

  try {
    switch (float_mat.channels()) {
      case 1: {
        int64_t step = float_mat.step / sizeof(float);
        std::vector<int64_t> strides = {step, 1};
        auto mat_tensor = torch::from_blob(
            float_mat.data,
            /*sizes=*/{float_mat.rows, float_mat.cols}, strides, deleter,
            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
        tensor = mat_tensor.clone();
      } break;

      case 3: {
        int64_t step = float_mat.step / sizeof(float);
        std::vector<int64_t> strides = {
            step, static_cast<int64_t>(float_mat.channels()), 1};
        auto mat_tensor = torch::from_blob(
            float_mat.data,
            /*sizes=*/{float_mat.rows, float_mat.cols, float_mat.channels()},
            strides, deleter,
            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
        tensor = mat_tensor.clone().permute({2, 0, 1});
      } break;

      default:
        std::cerr << "The mat has " << float_mat.channels()
                  << " channels (unsupported)" << std::endl;
        return torch::empty(
            {0},
            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
    }

    // Only transfer to different device if needed
    if (device_type != torch::kCUDA) {
      try {
        return tensor.to(device_type, true).contiguous();
      } catch (const c10::Error& e) {
        std::cerr << "Device transfer error: " << e.what() << std::endl;
        // Fall back to current device
        return tensor.contiguous();
      }
    }

    return tensor.contiguous();

  } catch (const std::exception& e) {
    std::cerr << "Exception in GPU tensor conversion: " << e.what()
              << std::endl;
    std::cerr << "GpuMat info - rows: " << float_mat.rows
              << ", cols: " << float_mat.cols
              << ", channels: " << float_mat.channels()
              << ", type: " << float_mat.type() << std::endl;
    return torch::empty(
        {0},
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
  }
}

/**
 * @brief
 *
 * @param torch::Tensor {channels, rows, cols}
 * @return cv::cuda::GpuMat {rows, cols, channels}
 */
inline cv::cuda::GpuMat torchTensor2CvGpuMat_Float32(torch::Tensor& tensor) {
  cv::cuda::GpuMat mat;
  torch::Tensor mat_tensor = tensor.clone();

  switch (mat_tensor.ndimension()) {
    case 2: {
      mat = cv::cuda::GpuMat(/*rows=*/mat_tensor.size(0),
                             /*cols=*/mat_tensor.size(1),
                             /*type=*/CV_32FC1,
                             /*data=*/mat_tensor.data_ptr<float>());
    } break;

    case 3: {
      mat_tensor = mat_tensor.detach().permute({1, 2, 0}).contiguous();
      mat = cv::cuda::GpuMat(/*rows=*/mat_tensor.size(0),
                             /*cols=*/mat_tensor.size(1),
                             /*type=*/CV_32FC3,
                             /*data=*/mat_tensor.data_ptr<float>());
    } break;

    default:
      std::cerr << "The tensor has unsupported number of channels!"
                << std::endl;
      break;
  }

  return mat.clone();
}

/**
 * @brief
 *
 * @param Eigen::Matrix column-major
 * @param torch::DeviceType
 * @return torch::Tensor row-major
 */
inline torch::Tensor EigenMatrix2TorchTensor(
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix,
    torch::DeviceType device_type = torch::kCUDA) {
  auto eigen_matrix_T = eigen_matrix;
  eigen_matrix_T.transposeInPlace();
  torch::Tensor tensor =
      torch::from_blob(
          /*data=*/eigen_matrix_T.data(),
          /*sizes=*/{eigen_matrix.rows(), eigen_matrix.cols()},
          /*options=*/torch::TensorOptions().dtype(torch::kFloat))
          .clone();

  tensor = tensor.to(device_type);
  return tensor;
}

}  // namespace tensor_utils

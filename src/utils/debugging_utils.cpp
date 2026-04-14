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

#include "utils/debugging_utils.h"

void visualizePointCloud(const torch::Tensor& points3D,
                         const torch::Tensor& colors,
                         const std::string& output_path) {
  // Convert tensors to CPU if they're on GPU
  auto points_cpu = points3D.cpu();
  auto colors_cpu = colors.cpu();

  // Open file for writing
  std::ofstream ply_file(output_path);
  if (!ply_file.is_open()) {
    throw std::runtime_error("Failed to open file for writing point cloud");
  }

  // Write PLY header
  int num_points = points_cpu.size(0);
  ply_file << "ply\n";
  ply_file << "format ascii 1.0\n";
  ply_file << "element vertex " << num_points << "\n";
  ply_file << "property float x\n";
  ply_file << "property float y\n";
  ply_file << "property float z\n";
  ply_file << "property uchar red\n";
  ply_file << "property uchar green\n";
  ply_file << "property uchar blue\n";
  ply_file << "end_header\n";

  // Write points and colors
  auto points_accessor = points_cpu.accessor<float, 2>();
  auto colors_accessor = colors_cpu.accessor<float, 2>();

  for (int i = 0; i < num_points; i++) {
    // Write XYZ coordinates
    ply_file << points_accessor[i][0] << " " << points_accessor[i][1] << " "
             << points_accessor[i][2] << " ";

    // Convert float RGB values [0,1] to uchar [0,255] and write
    int r = static_cast<int>(colors_accessor[i][0] * 255);
    int g = static_cast<int>(colors_accessor[i][1] * 255);
    int b = static_cast<int>(colors_accessor[i][2] * 255);
    ply_file << r << " " << g << " " << b << "\n";
  }

  ply_file.close();
  std::cout << "Point cloud saved to " << output_path << std::endl;
  std::cout << "Number of points saved: " << num_points << std::endl;
}

void saveColorizedDepthMap(const torch::Tensor& depth,
                           int height,
                           int width,
                           const std::string& filepath,
                           float min_depth,
                           float max_depth) {
  try {
    torch::NoGradGuard no_grad;

    // Print tensor info for debugging
    std::cout << "Depth tensor shape: " << depth.sizes() << std::endl;
    std::cout << "Min value: " << torch::min(depth).item<float>() << std::endl;
    std::cout << "Max value: " << torch::max(depth).item<float>() << std::endl;

    // Reshape flattened depth tensor to 2D using provided dimensions
    auto depth_2d = depth.reshape({height, width});

    // Clamp values to valid range
    depth_2d = torch::clamp(depth_2d, min_depth, max_depth);

    // Normalize to 0-1 range
    depth_2d = (depth_2d - min_depth) / (max_depth - min_depth);

    // Convert to CPU and uint8
    auto depth_cpu = (depth_2d * 255).to(torch::kCPU).to(torch::kUInt8);

    // Convert to OpenCV Mat
    cv::Mat depth_mat(height, width, CV_8UC1);
    std::memcpy(depth_mat.data, depth_cpu.data_ptr(),
                depth_cpu.numel() * sizeof(uint8_t));

    // Apply colormap
    cv::Mat colored_depth;
    cv::applyColorMap(depth_mat, colored_depth, cv::COLORMAP_TURBO);

    // Save the colorized depth map using the provided filepath
    bool success = cv::imwrite(filepath, colored_depth);
    if (!success) {
      std::cerr << "Failed to save depth map to: " << filepath << std::endl;
    } else {
      std::cout << "Successfully saved depth map to: " << filepath << std::endl;
    }

  } catch (const std::exception& e) {
    std::cerr << "Error in saveColorizedDepthMap: " << e.what() << std::endl;
  }
}

/**
 * Colorizes a depth tensor and saves it to a file
 * @param depth_tensor The input depth tensor (1 x H x W)
 * @param output_path Path where the colorized image will be saved
 * @param min_depth Optional minimum depth value for normalization
 * @param max_depth Optional maximum depth value for normalization
 * @param colormap OpenCV colormap type (default: COLORMAP_JET)
 * @return true if successful, false otherwise
 */
bool colorize_and_save_depth(const torch::Tensor& depth_tensor,
                             const std::string& output_path,
                             float min_depth,
                             float max_depth,
                             int colormap) {
  try {
    // Ensure tensor is on CPU and get dimensions
    auto depth = depth_tensor.cpu();

    // Handle different tensor shapes
    if (depth.dim() == 4 && depth.size(0) == 1) {  // N x C x H x W with N=1
      depth = depth.squeeze(0);
    }

    if (depth.dim() == 3 && depth.size(0) == 1) {  // C x H x W with C=1
      depth = depth.squeeze(0);
    }

    // Ensure we have a 2D tensor now
    if (depth.dim() != 2) {
      std::cerr << "Error: Depth tensor must be 2D after squeezing, but got "
                   "dimensions: "
                << depth.dim() << std::endl;
      return false;
    }

    // Get tensor dimensions
    int height = depth.size(0);
    int width = depth.size(1);

    // Clone and normalize the depth values between 0 and 1
    auto normalized = depth.clone();

    // Auto-detect min/max if not provided
    if (min_depth < 0 || max_depth < 0) {
      min_depth = depth.min().item<float>();
      max_depth = depth.max().item<float>();
    }

    normalized = torch::clamp(normalized, min_depth, max_depth);

    // Handle case where min == max (flat depth)
    if (std::abs(max_depth - min_depth) < 1e-6) {
      normalized.fill_(0.5f);
    } else {
      // Normalize to 0-1 range
      normalized = (normalized - min_depth) / (max_depth - min_depth);
    }

    // Invert so close objects are bright, far objects are dark
    normalized = 1.0f - normalized;

    // Convert to OpenCV format (0-255 uint8)
    normalized = normalized * 255.0f;
    auto depth_cv = cv::Mat(height, width, CV_8UC1);

    // Copy data from tensor to OpenCV mat
    auto accessor = normalized.accessor<float, 2>();
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        depth_cv.at<uchar>(y, x) = static_cast<uchar>(accessor[y][x]);
      }
    }

    // Apply colormap
    cv::Mat colored;
    cv::applyColorMap(depth_cv, colored, colormap);

    // Save image
    return cv::imwrite(output_path, colored);
  } catch (const std::exception& e) {
    std::cerr << "Error in colorize_and_save_depth: " << e.what() << std::endl;
    return false;
  }
}
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

#include <cuda_runtime.h>
#include <torch/torch.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

void visualizePointCloud(const torch::Tensor &points3D,
                         const torch::Tensor &colors,
                         const std::string &output_path = "point_cloud.ply");

void saveColorizedDepthMap(const torch::Tensor &depth,
                           int height,
                           int width,
                           const std::string &filepath,
                           float min_depth = 0.0f,
                           float max_depth = 20.0f);

bool colorize_and_save_depth(const torch::Tensor &depth_tensor,
                             const std::string &output_path,
                             float min_depth = -1.0f,
                             float max_depth = -1.0f,
                             int colormap = cv::COLORMAP_MAGMA);
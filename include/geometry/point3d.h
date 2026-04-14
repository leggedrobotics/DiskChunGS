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

#include <Eigen/Core>

class Point3D {
 public:
  Point3D()
      : xyz_(0.0, 0.0, 0.0),
        color_(0.0f, 0.0f, 0.0f),
        color256_(0, 0, 0),
        error_(-1.0) {}

 public:
  Eigen::Vector3d xyz_;

  Eigen::Matrix<uint8_t, 3, 1>
      color256_;  // not needed if we get color_ directly
  Eigen::Matrix<float, 3, 1> color_;

  double error_;
};

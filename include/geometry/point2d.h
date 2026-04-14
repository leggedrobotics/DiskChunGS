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
#include <limits>

#include "types.h"

class Point2D {
 public:
  Point2D()
      : xy_(0.0, 0.0), point3D_id_(std::numeric_limits<std::uint64_t>::max()) {}

 public:
  Eigen::Vector2d xy_;

  std::uint64_t point3D_id_;
};

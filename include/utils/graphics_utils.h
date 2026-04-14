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

#include "gaussian_splatting/utils/graphics_utils.h"

#include <Eigen/Geometry>

namespace graphics_utils {

inline int roundToIntegerMultipleOf16(int integer) {
  int remainder = integer % 16;

  if (remainder == 0) {
    return integer;
  } else if (remainder < 8) {
    return integer - remainder;
  } else {
    return integer - remainder + 16;
  }

  return integer;
}

}  // namespace graphics_utils

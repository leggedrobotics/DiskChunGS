
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

#include <ros/ros.h>

#include "wrapper.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "diskchungs_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  try {
    GaussianSLAMWrapper gaussian_slam(nh, pnh);
    ros::spin();
    ROS_INFO("Gaussian SLAM node shutting down");
  } catch (const std::exception& e) {
    ROS_ERROR("Exception in Gaussian SLAM node: %s", e.what());
    return 1;
  }

  return 0;
}
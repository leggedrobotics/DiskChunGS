
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

#include <cv_bridge/cv_bridge.h>
#include <dlfcn.h>
#include <geometry_msgs/TransformStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <torch/torch.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <filesystem>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <string>
#include <thread>
#include <vector>

#include "ORB-SLAM3/include/System.h"
#include "gaussian_mapper.h"
#include "viewer/imgui_viewer.h"

// Configuration structure for GaussianSLAMWrapper
struct WrapperConfig {
  // File paths
  std::string vocabulary_path;
  std::string orb_settings_path;
  std::string gaussian_settings_path;
  std::string output_directory;

  // Topic names
  std::string image_topic;  // primary camera (left for stereo, rgb for rgbd, sole camera for mono)
  std::string right_topic;
  std::string depth_topic;

  // Frame names for external pose mode
  std::string target_frame;
  std::string source_frame;

  // Operating modes
  std::string mode;       // stereo, mono, rgbd
  std::string slam_mode;  // orbslam, external

  // Settings
  bool use_viewer;
  double timeout_duration;

  // Load configuration from ROS parameter server
  static WrapperConfig loadFromROS(ros::NodeHandle& pnh);
};

class GaussianSLAMWrapper {
 public:
  GaussianSLAMWrapper(ros::NodeHandle &nh, ros::NodeHandle &pnh);
  ~GaussianSLAMWrapper();

 private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  // Topic subscribers
  message_filters::Subscriber<sensor_msgs::Image> image_sub_;
  message_filters::Subscriber<sensor_msgs::Image> right_sub_;
  message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
  ros::Subscriber mono_sub_;

  // Synchronization
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                          sensor_msgs::Image>
      sync_pol;
  std::shared_ptr<message_filters::Synchronizer<sync_pol>> sync_;
  std::shared_ptr<message_filters::Synchronizer<sync_pol>> rgbd_sync_;

  // TF listener for external poses
  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener;
  bool first_frame = true;
  Sophus::SE3f T_init;

  // SLAM system
  std::shared_ptr<ORB_SLAM3::System> slam_system_;
  std::shared_ptr<GaussianMapper> gaussian_mapper_;
  std::shared_ptr<ImGuiViewer> viewer_;

  // Threads
  std::thread mapper_thread_;
  std::thread viewer_thread_;

  // Configuration
  WrapperConfig config_;

  // Timeout-related members
  ros::Timer timeout_timer_;        // Timer to check for timeouts
  ros::Time last_callback_time_;    // Time of the last callback
  bool data_started_ = false;       // Flag to track if data has started
  bool stopped_ = false;            // Flag to track if we've already stopped
  std::mutex
      timeout_mutex_;  // Mutex for thread-safe access to timeout variables
  std::atomic<bool> mapping_completed_{false};
  ros::Timer status_check_timer_;

  // Callback methods
  void monoCallback(const sensor_msgs::ImageConstPtr &msg);
  void stereoCallback(const sensor_msgs::ImageConstPtr &left,
                      const sensor_msgs::ImageConstPtr &right);
  void rgbdCallback(const sensor_msgs::ImageConstPtr &rgb,
                    const sensor_msgs::ImageConstPtr &depth);
  void timeoutCallback(const ros::TimerEvent &event);
  void checkMappingStatus(const ros::TimerEvent &event);

  // Helper methods
  void initializeSLAMSystem();
  void initializeGaussianMapper();
  bool getExternalPose(Sophus::SE3f &pose, double timestamp);
  void updateCallbackTime();  // Helper to update the last callback time
};
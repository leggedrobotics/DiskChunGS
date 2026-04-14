
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

#include "wrapper.h"

WrapperConfig WrapperConfig::loadFromROS(ros::NodeHandle &pnh) {
  WrapperConfig config;

  // Load required parameters (throw if missing)
  if (!pnh.getParam("vocabulary_path", config.vocabulary_path)) {
    throw std::runtime_error("Failed to load vocabulary_path parameter");
  }
  if (!pnh.getParam("orb_settings_path", config.orb_settings_path)) {
    throw std::runtime_error("Failed to load orb_settings_path parameter");
  }
  if (!pnh.getParam("gaussian_settings_path", config.gaussian_settings_path)) {
    throw std::runtime_error("Failed to load gaussian_settings_path parameter");
  }
  if (!pnh.getParam("output_directory", config.output_directory)) {
    throw std::runtime_error("Failed to load output_directory parameter");
  }

  // Load optional parameters with defaults
  pnh.param<bool>("use_viewer", config.use_viewer, false);
  pnh.param<std::string>("mode", config.mode, "stereo");
  pnh.param<std::string>("slam_mode", config.slam_mode, "orbslam");
  pnh.param<std::string>("image_topic", config.image_topic,
                         "/camera/image_raw");
  if (config.mode == "stereo") {
    pnh.param<std::string>("right_topic", config.right_topic,
                           "/camera/right/image_raw");
  } else if (config.mode == "rgbd") {
    pnh.param<std::string>("depth_topic", config.depth_topic,
                           "/camera/depth/image_raw");
  }
  pnh.param<std::string>("target_frame", config.target_frame, "map");
  pnh.param<std::string>("source_frame", config.source_frame,
                         "zed2i_left_camera_frame");
  pnh.param<double>("timeout_duration", config.timeout_duration, 20.0);

  // Validate mode combinations
  if (config.slam_mode == "external" && config.mode == "mono") {
    throw std::runtime_error(
        "External slam_mode is not supported with mono mode");
  }

  // Verify file paths exist
  if (!std::filesystem::exists(config.vocabulary_path)) {
    throw std::runtime_error("Vocabulary file not found: " +
                             config.vocabulary_path);
  }
  if (!std::filesystem::exists(config.orb_settings_path)) {
    throw std::runtime_error("ORB settings file not found: " +
                             config.orb_settings_path);
  }
  if (!std::filesystem::exists(config.gaussian_settings_path)) {
    throw std::runtime_error("Gaussian settings file not found: " +
                             config.gaussian_settings_path);
  }

  // Log configuration
  ROS_INFO("Configuration loaded:");
  ROS_INFO("  mode: %s", config.mode.c_str());
  ROS_INFO("  slam_mode: %s", config.slam_mode.c_str());
  ROS_INFO("  vocabulary_path: %s", config.vocabulary_path.c_str());
  ROS_INFO("  orb_settings_path: %s", config.orb_settings_path.c_str());
  ROS_INFO("  gaussian_settings_path: %s",
           config.gaussian_settings_path.c_str());
  ROS_INFO("  output_directory: %s", config.output_directory.c_str());
  ROS_INFO("  timeout_duration: %.1f seconds", config.timeout_duration);
  ROS_INFO("  use_viewer: %d", config.use_viewer);

  if (config.slam_mode == "external") {
    ROS_INFO("  target_frame: %s", config.target_frame.c_str());
    ROS_INFO("  source_frame: %s", config.source_frame.c_str());
  }

  return config;
}

GaussianSLAMWrapper::GaussianSLAMWrapper(ros::NodeHandle &nh,
                                         ros::NodeHandle &pnh)
    : nh_(nh),
      pnh_(pnh),
      tfListener(tfBuffer),
      config_(WrapperConfig::loadFromROS(pnh)) {
  ROS_INFO("GaussianSLAMWrapper constructor starting...");

  timeout_timer_ = nh_.createTimer(ros::Duration(1.0),
                                   &GaussianSLAMWrapper::timeoutCallback, this);
  status_check_timer_ = nh_.createTimer(
      ros::Duration(1.0), &GaussianSLAMWrapper::checkMappingStatus, this);

  // Initialize subscribers based on mode
  if (config_.mode == "stereo") {
    image_sub_.subscribe(nh_, config_.image_topic, 1);
    right_sub_.subscribe(nh_, config_.right_topic, 1);
    sync_.reset(new message_filters::Synchronizer<sync_pol>(
        sync_pol(30), image_sub_, right_sub_));
  } else if (config_.mode == "rgbd") {
    image_sub_.subscribe(nh_, config_.image_topic, 1);
    depth_sub_.subscribe(nh_, config_.depth_topic, 1);
    rgbd_sync_.reset(new message_filters::Synchronizer<sync_pol>(
        sync_pol(30), image_sub_, depth_sub_));
  }

  if (config_.slam_mode == "orbslam") {
    initializeSLAMSystem();
  }

  initializeGaussianMapper();

  // Register appropriate callback based on mode
  if (config_.mode == "stereo") {
    ROS_INFO("Registering stereo callback...");
    ROS_INFO("Left topic: %s", image_sub_.getTopic().c_str());
    ROS_INFO("Right topic: %s", right_sub_.getTopic().c_str());
    sync_->registerCallback(
        boost::bind(&GaussianSLAMWrapper::stereoCallback, this, _1, _2));
  } else if (config_.mode == "rgbd") {
    ROS_INFO("Registering RGB-D callback...");
    ROS_INFO("RGB topic: %s", image_sub_.getTopic().c_str());
    ROS_INFO("Depth topic: %s", depth_sub_.getTopic().c_str());
    rgbd_sync_->registerCallback(
        boost::bind(&GaussianSLAMWrapper::rgbdCallback, this, _1, _2));
  } else if (config_.mode == "mono") {
    ROS_INFO("Registering mono callback...");
    mono_sub_ = nh_.subscribe(config_.image_topic, 1,
                              &GaussianSLAMWrapper::monoCallback, this);
    ROS_INFO("Image topic: %s", config_.image_topic.c_str());
  } else {
    throw std::runtime_error("Invalid mode: " + config_.mode);
  }

  ROS_INFO("GaussianSLAMWrapper initialization complete!");
}

bool GaussianSLAMWrapper::getExternalPose(Sophus::SE3f &pose,
                                          double timestamp) {
  // First attempt with waitForTransform to block until the transform is
  // available
  if (!tfBuffer.canTransform(config_.target_frame, config_.source_frame,
                             ros::Time(timestamp), ros::Duration(0.5))) {
    ROS_WARN("Transform from %s to %s not available yet, waiting...",
             config_.source_frame.c_str(), config_.target_frame.c_str());
    return false;
  }
  try {
    // Now try to lookup the transform
    geometry_msgs::TransformStamped transformStamped = tfBuffer.lookupTransform(
        config_.target_frame, config_.source_frame, ros::Time(timestamp));

    // Get the rotation quaternion and translation
    Eigen::Quaternionf quat(transformStamped.transform.rotation.w,
                            transformStamped.transform.rotation.x,
                            transformStamped.transform.rotation.y,
                            transformStamped.transform.rotation.z);

    Eigen::Vector3f trans(transformStamped.transform.translation.x,
                          transformStamped.transform.translation.y,
                          transformStamped.transform.translation.z);

    // Twc: pose of camera in world frame
    Sophus::SE3f Twc_ros(quat, trans);

    // Normalize so that the first received frame is the world origin
    if (first_frame) {
      T_init = Twc_ros;
      first_frame = false;
    }
    Sophus::SE3f Twc_normalized = T_init.inverse() * Twc_ros;

    // Return Tcw (camera-to-world inverse) as expected by the mapper
    pose = Twc_normalized.inverse();

    return true;
  } catch (tf2::TransformException &ex) {
    ROS_WARN("%s", ex.what());
    return false;
  }
}

void GaussianSLAMWrapper::initializeSLAMSystem() {
  ROS_INFO("Initializing SLAM system...");
  try {
    ORB_SLAM3::System::eSensor system_mode;
    // Select the appropriate sensor mode
    if (config_.mode == "stereo") {
      system_mode = ORB_SLAM3::System::STEREO;
    } else if (config_.mode == "rgbd") {
      system_mode = ORB_SLAM3::System::RGBD;
    } else if (config_.mode == "mono") {
      system_mode = ORB_SLAM3::System::MONOCULAR;
    } else {
      throw std::runtime_error("Invalid mode specified: " + config_.mode);
    }

    slam_system_ = std::make_shared<ORB_SLAM3::System>(
        config_.vocabulary_path, config_.orb_settings_path, system_mode);
    ROS_INFO("SLAM system object created successfully");
  } catch (const std::exception &e) {
    ROS_ERROR("Exception during SLAM system initialization: %s", e.what());
    throw;
  }
}

void GaussianSLAMWrapper::initializeGaussianMapper() {
  ROS_INFO("Creating Gaussian Mapper...");

  if (config_.slam_mode == "external") {
    ORB_SLAM3::System::eSensor sensor_type;

    if (config_.mode == "mono") {
      sensor_type = ORB_SLAM3::System::MONOCULAR;
    } else if (config_.mode == "stereo") {
      sensor_type = ORB_SLAM3::System::STEREO;
    } else if (config_.mode == "rgbd") {
      sensor_type = ORB_SLAM3::System::RGBD;
    } else {
      throw std::runtime_error("[Gaussian Mapper]Unsupported sensor type!");
    }

    gaussian_mapper_ = std::make_shared<GaussianMapper>(
        slam_system_, config_.gaussian_settings_path, config_.output_directory,
        0, torch::kCUDA, sensor_type, config_.orb_settings_path);

    gaussian_mapper_->setCompletionCallback(
        [this]() { this->mapping_completed_.store(true); });

    mapper_thread_ = std::thread(&GaussianMapper::run_external_poses,
                                 gaussian_mapper_.get());

  } else if (config_.slam_mode == "orbslam") {
    gaussian_mapper_ = std::make_shared<GaussianMapper>(
        slam_system_, config_.gaussian_settings_path, config_.output_directory,
        0,            // stream id
        torch::kCUDA  // assuming CUDA is available
    );

    mapper_thread_ = std::thread(&GaussianMapper::run, gaussian_mapper_.get());
  }

  if (config_.use_viewer) {
    ROS_INFO("Initializing viewer...");
    viewer_ =
        std::make_shared<ImGuiViewer>(slam_system_, gaussian_mapper_, true,
                                      (config_.slam_mode == "external"));
    viewer_thread_ = std::thread(&ImGuiViewer::run, viewer_.get());
    ROS_INFO("Viewer initialized and started.");
  }
}

void GaussianSLAMWrapper::monoCallback(const sensor_msgs::ImageConstPtr &msg) {
  updateCallbackTime();
  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    if (msg->encoding == "bayer_rggb8") {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
    } else {
      cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::RGB8);
    }

    double timestamp = msg->header.stamp.toSec();

    if (config_.slam_mode == "external") {
      ROS_ERROR("Mono doesn't support external mode right now!");
      return;
    }

    slam_system_->TrackMonocular(cv_ptr->image, timestamp,
                                 std::vector<ORB_SLAM3::IMU::Point>(),
                                 std::to_string(msg->header.seq));
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  } catch (const std::exception &e) {
    ROS_ERROR("Exception in TrackMonocular: %s", e.what());
  }
}

void GaussianSLAMWrapper::stereoCallback(
    const sensor_msgs::ImageConstPtr &msg_left,
    const sensor_msgs::ImageConstPtr &msg_right) {
  updateCallbackTime();

  cv_bridge::CvImageConstPtr cv_left, cv_right;
  try {
    // If images are Bayer RGGB8, convert to RGB8
    if (msg_left->encoding == "bayer_rggb8") {
      cv_left =
          cv_bridge::toCvCopy(msg_left, sensor_msgs::image_encodings::RGB8);
      cv_right =
          cv_bridge::toCvCopy(msg_right, sensor_msgs::image_encodings::RGB8);
    } else {
      cv_left =
          cv_bridge::toCvShare(msg_left, sensor_msgs::image_encodings::RGB8);
      cv_right =
          cv_bridge::toCvShare(msg_right, sensor_msgs::image_encodings::RGB8);
    }

  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  // Get timestamp from message
  double timestamp = msg_left->header.stamp.toSec();

  try {
    if (config_.slam_mode == "external") {
      Sophus::SE3f Tcw;
      if (getExternalPose(Tcw, timestamp)) {
        gaussian_mapper_->handleNewFrameExternal(
            cv_left->image, cv_right->image, Tcw, timestamp);
      }
    } else if (config_.slam_mode == "orbslam") {
      if (!slam_system_) {
        ROS_ERROR("SLAM system pointer is null!");
        return;
      }

      try {
        slam_system_->TrackStereo(cv_left->image, cv_right->image, timestamp,
                                  std::vector<ORB_SLAM3::IMU::Point>(),
                                  std::to_string(msg_left->header.seq));
      } catch (const std::exception &e) {
        ROS_ERROR("Exception in TrackStereo: %s", e.what());
      }
    }
  } catch (const std::exception &e) {
    ROS_ERROR("Exception in TrackStereo: %s", e.what());
  }
}

void GaussianSLAMWrapper::rgbdCallback(
    const sensor_msgs::ImageConstPtr &msg_rgb,
    const sensor_msgs::ImageConstPtr &msg_depth) {
  updateCallbackTime();
  cv_bridge::CvImageConstPtr cv_rgb, cv_depth;
  try {
    // Convert RGB image (existing code)
    if (msg_rgb->encoding == "bayer_rggb8") {
      cv_rgb = cv_bridge::toCvCopy(msg_rgb, sensor_msgs::image_encodings::RGB8);
    } else {
      cv_rgb =
          cv_bridge::toCvShare(msg_rgb, sensor_msgs::image_encodings::RGB8);
    }

    // Handle depth image (existing code)
    cv::Mat depth_converted;
    if (msg_depth->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
      cv_depth = cv_bridge::toCvShare(msg_depth,
                                      sensor_msgs::image_encodings::TYPE_32FC1);
      // Convert from meters to millimeters
      cv_depth->image.convertTo(depth_converted, CV_16UC1, 1000);
      // Replace inf/nan with 0
      depth_converted.setTo(0, cv_depth->image <= 0);
    } else {
      // Assume 16UC1 and use as is
      cv_depth = cv_bridge::toCvShare(msg_depth,
                                      sensor_msgs::image_encodings::TYPE_16UC1);
      depth_converted = cv_depth->image;
    }

    // Get timestamp from message
    double timestamp = msg_rgb->header.stamp.toSec();

    // Tracking logic
    try {
      if (config_.slam_mode == "external") {
        Sophus::SE3f Tcw;
        if (getExternalPose(Tcw, timestamp)) {
          gaussian_mapper_->handleNewFrameExternal(
              cv_rgb->image, cv_depth->image, Tcw, timestamp);
        }
      } else if (config_.slam_mode == "orbslam") {
        if (!slam_system_) {
          ROS_ERROR("SLAM system pointer is null!");
          return;
        }

        slam_system_->TrackRGBD(cv_rgb->image, depth_converted, timestamp, {},
                                std::to_string(msg_rgb->header.seq));
      }
    } catch (const std::exception &e) {
      ROS_ERROR("Exception in TrackRGBD: %s", e.what());
    }

  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
}
void GaussianSLAMWrapper::timeoutCallback(const ros::TimerEvent &event) {
  std::lock_guard<std::mutex> lock(timeout_mutex_);

  // Only check for timeout if we've started receiving data and haven't already
  // stopped
  if (data_started_ && !stopped_) {
    ros::Duration elapsed = ros::Time::now() - last_callback_time_;

    if (elapsed.toSec() > config_.timeout_duration) {
      ROS_INFO(
          "No callbacks received for %.1f seconds, signaling data stream "
          "stopped",
          elapsed.toSec());

      // Signal to the gaussian mapper that the external data has stopped
      if (gaussian_mapper_) {
        ROS_INFO("Calling signalExternalDataStopped on gaussian mapper");
        gaussian_mapper_->signalExternalDataStopped();
        stopped_ = true;
        ROS_INFO("Called signalExternalDataStopped on gaussian mapper");
      }
    }
  }
}

// Add a helper method to update the last callback time
void GaussianSLAMWrapper::updateCallbackTime() {
  std::lock_guard<std::mutex> lock(timeout_mutex_);
  last_callback_time_ = ros::Time::now();
  if (!data_started_) {
    data_started_ = true;
    ROS_INFO("Data stream has started, timeout monitoring active");
  }
}

void GaussianSLAMWrapper::checkMappingStatus(const ros::TimerEvent &event) {
  // Check if mapper has signaled completion
  if (mapping_completed_) {
    ROS_INFO("Mapping process complete, shutting down node.");
    ros::shutdown();
  }
}

GaussianSLAMWrapper::~GaussianSLAMWrapper() {
  // Stop the timeout timer
  timeout_timer_.stop();

  std::cout << "Shutting down GaussianSLAMWrapper..." << std::endl;

  if (slam_system_) {
    slam_system_->Shutdown();
  }

  if (mapper_thread_.joinable()) {
    mapper_thread_.join();
  }

  if (viewer_thread_.joinable()) {
    viewer_thread_.join();
  }
}
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

#include "gaussian_mapper.h"
#include "gaussian_mapper_external.h"

// ============================================================================
// Frame and LeakyFrameQueue implementations
// ============================================================================

Frame::Frame(const cv::Mat& rgb,
             const cv::Mat& depth,
             const Sophus::SE3f& p,
             double ts)
    : rgb_image(rgb.clone()),
      depth_image(depth.clone()),
      pose(p),
      timestamp(ts) {}

LeakyFrameQueue::LeakyFrameQueue(size_t max_size) : max_size_(max_size) {}

void LeakyFrameQueue::push(Frame&& frame) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (queue_.size() >= max_size_) {
    return;
  }

  queue_.push_back(std::move(frame));
  cv_.notify_one();
}

std::optional<Frame> LeakyFrameQueue::pop(bool wait) {
  std::unique_lock<std::mutex> lock(mutex_);

  if (wait) {
    cv_.wait(lock, [this] { return !queue_.empty() || stopped_; });
  }

  if (queue_.empty() || stopped_) {
    return std::nullopt;
  }

  Frame frame = std::move(queue_.front());
  queue_.pop_front();
  return frame;
}

void LeakyFrameQueue::stop() {
  std::lock_guard<std::mutex> lock(mutex_);
  stopped_ = true;
  cv_.notify_all();
}

bool LeakyFrameQueue::empty() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.empty();
}

size_t LeakyFrameQueue::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.size();
}

// ============================================================================
// GaussianMapper external pose mode methods
// ============================================================================

void GaussianMapper::handleNewFrameExternal(const cv::Mat& rgb_image,
                                            const cv::Mat& depth_or_right_image,
                                            const Sophus::SE3f& pose,
                                            const double timestamp) {
  frame_queue_.push(Frame(rgb_image, depth_or_right_image, pose, timestamp));
}

void GaussianMapper::run_external_poses() {
  training_start_time_ = std::chrono::steady_clock::now();

  // Initialize output directories
  std::filesystem::remove_all(chunk_save_dir_);
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(chunk_save_dir_)

  std::filesystem::remove_all(keyframe_save_dir_);
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(keyframe_save_dir_)

  // Set default scene extent for external mode
  scene_->cameras_extent_ = 1.0f;

  // Initial mapping: process first keyframe and setup training
  while (!initial_mapped_ && !isStopped() && !isExternalDataStopped()) {
    auto maybe_frame = frame_queue_.pop(true);
    if (!maybe_frame) {
      continue;
    }

    auto& frame = *maybe_frame;
    handleNewKeyframeFromExternal(frame.rgb_image, frame.depth_image,
                                  frame.pose, frame.timestamp);

    std::unique_lock<std::mutex> lock_render(mutex_render_);
    gaussians_->trainingSetup(opt_params_);
    initial_mapped_ = true;
  }

  // Main training loop: process incoming frames and train
  while (!isExternalDataStopped() && !isStopped()) {
    while (auto maybe_frame = frame_queue_.pop(false)) {
      handleNewKeyframeFromExternal(maybe_frame->rgb_image,
                                    maybe_frame->depth_image, maybe_frame->pose,
                                    maybe_frame->timestamp);
    }

    trainForOneIteration();
  }

  frame_queue_.stop();

  // Save final results
  saveTotalGaussians("_shutdown");
  renderAndRecordAllKeyframes("_shutdown");
  saveScene(result_dir_ / (std::to_string(getIteration()) + "_shutdown") /
            "data");
  writeKeyframeUsedTimes(result_dir_ / "used_times", "final");
  writeTrainingMetricsCSV(result_dir_);

  signalStop();
  if (completion_callback_) {
    completion_callback_();
  }
}

bool GaussianMapper::isKeyframe(const Sophus::SE3f& current_pose,
                                double current_time) {
  if (scene_->keyframes().empty()) {
    return true;
  }

  // Check temporal threshold
  if (current_time - last_keyframe_timestamp_ < min_keyframe_time_) {
    return false;
  }

  // Check motion thresholds (translation and rotation)
  Sophus::SE3f relative_motion = current_pose.inverse() * last_keyframe_pose_;
  float translation = relative_motion.translation().norm();
  float rotation = Eigen::AngleAxisf(relative_motion.rotationMatrix()).angle();

  if (translation > min_keyframe_translation_ ||
      rotation > min_keyframe_rotation_) {
    last_keyframe_pose_ = current_pose;
    return true;
  }

  return false;
}

void GaussianMapper::handleNewKeyframeFromExternal(
    cv::Mat& rgb_image,
    cv::Mat& depth_or_right_image,
    const Sophus::SE3f& pose,
    const double timestamp) {
  setRecentExternalData(rgb_image, pose);

  if (!isKeyframe(pose, timestamp)) {
    return;
  }

  last_keyframe_pose_ = pose;
  last_keyframe_timestamp_ = timestamp;

  std::shared_ptr<GaussianKeyframe> pkf = std::make_shared<GaussianKeyframe>(
      scene_->keyframes().size(), getIteration(), keyframe_save_dir_);

  pkf->setPose(pose.unit_quaternion().cast<double>(),
               pose.translation().cast<double>());

  // External mode uses a single camera (index 0)
  Camera& camera = scene_->cameras_.at(0);
  createAndInitializeKeyframe(pkf, rgb_image, depth_or_right_image, camera);
}

void GaussianMapper::setRecentExternalData(const cv::Mat& rgb_image,
                                           const Sophus::SE3f& pose) {
  std::unique_lock<std::mutex> lock(mutex_external_data_);

  external_image_ = rgb_image.clone();
  external_pose_ = pose;
}

std::tuple<const cv::Mat, const Sophus::SE3f>
GaussianMapper::getRecentExternalData() {
  std::unique_lock<std::mutex> lock(mutex_external_data_);

  return std::make_tuple(external_image_, external_pose_);
}

void GaussianMapper::setCompletionCallback(std::function<void()> callback) {
  completion_callback_ = callback;
}

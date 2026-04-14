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

#include <condition_variable>
#include <deque>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <optional>

#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"

/**
 * @brief Frame structure for external pose mode
 *
 * Encapsulates a single frame with pose, images, and timestamp for external
 * tracking systems that provide camera poses directly (without ORB-SLAM).
 */
struct Frame {
  cv::Mat rgb_image;    ///< RGB image data
  cv::Mat depth_image;  ///< Depth map or right stereo image
  Sophus::SE3f pose;    ///< Camera pose (SE3 transformation)
  double timestamp;     ///< Frame timestamp

  /**
   * @brief Construct a new Frame
   * @param rgb RGB image
   * @param depth Depth map or right stereo image
   * @param p Camera pose
   * @param ts Timestamp
   */
  Frame(const cv::Mat& rgb,
        const cv::Mat& depth,
        const Sophus::SE3f& p,
        double ts);
};

/**
 * @brief Thread-safe leaky frame queue for external pose mode
 *
 * A bounded FIFO queue that drops incoming frames when full (leaky behavior)
 * to prevent unbounded memory growth when processing cannot keep up with
 * frame ingestion rate.
 */
class LeakyFrameQueue {
 public:
  /**
   * @brief Construct a new leaky frame queue
   * @param max_size Maximum queue capacity before dropping frames (default: 20)
   */
  explicit LeakyFrameQueue(size_t max_size = 20);

  /**
   * @brief Push a frame onto the queue (drops if full)
   * @param frame Frame to push (moved into queue)
   */
  void push(Frame&& frame);

  /**
   * @brief Pop a frame from the queue
   * @param wait If true, blocks until frame available; if false, returns
   * immediately
   * @return Frame if available, std::nullopt otherwise
   */
  std::optional<Frame> pop(bool wait = true);

  /**
   * @brief Signal queue to stop and unblock waiting threads
   */
  void stop();

  /**
   * @brief Check if queue is empty
   * @return True if empty, false otherwise
   */
  bool empty() const;

  /**
   * @brief Get current queue size
   * @return Number of frames in queue
   */
  size_t size() const;

 private:
  std::deque<Frame> queue_;     ///< Internal frame storage
  mutable std::mutex mutex_;    ///< Synchronization mutex
  std::condition_variable cv_;  ///< Condition variable for blocking pop
  const size_t max_size_;       ///< Maximum queue capacity
  bool stopped_{false};         ///< Stop flag for graceful shutdown
};

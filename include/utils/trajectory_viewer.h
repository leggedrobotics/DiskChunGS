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

#include <deque>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>

// Forward declaration
class GaussianMapper;

class TrajectoryViewer {
 public:
  explicit TrajectoryViewer(GaussianMapper* pGausMapper);

  void run();

  bool isStopped();
  void signalStop(const bool going_to_stop = true);

  // Record a keyframe selection for visualization
  void recordKeyframeSelection(int keyframe_id);

 private:
  GaussianMapper* pGausMapper_;

  bool stopped_;
  std::mutex mutex_status_;

  int window_size_ = 800;
  int margin_ = 50;

  // Selection tracking
  std::deque<int> recent_selected_keyframe_ids_;
  std::mutex selection_mutex_;
  static constexpr size_t max_selection_history_ = 100;
};

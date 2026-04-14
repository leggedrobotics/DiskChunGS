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

#include "utils/trajectory_viewer.h"

#include <iostream>
#include <unordered_set>
#include <vector>

#include "gaussian_mapper.h"

TrajectoryViewer::TrajectoryViewer(GaussianMapper* pGausMapper)
    : pGausMapper_(pGausMapper), stopped_(false) {}

void TrajectoryViewer::run() {
  std::string window_name = "Trajectory Top-Down View";
  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

  while (!isStopped()) {
    // Create white background
    cv::Mat display(window_size_, window_size_, CV_8UC3,
                    cv::Scalar(255, 255, 255));

    // Get keyframes from scene
    if (pGausMapper_ && pGausMapper_->scene_) {
      auto& keyframes_map = pGausMapper_->scene_->keyframes();

      if (!keyframes_map.empty()) {
        // Struct to hold keyframe data for visualization
        struct KeyframeData {
          float x, z;  // Position in top-down view
          Eigen::Matrix3f rotation;
          float fov_x;
          float zfar;
        };

        // Collect data from last 100 keyframes
        std::vector<std::pair<float, float>> positions;
        std::vector<KeyframeData> keyframe_data;

        auto it = keyframes_map.rbegin();
        int count = 0;
        for (; it != keyframes_map.rend() && count < 3000; ++it, ++count) {
          auto kf = it->second;
          if (kf) {
            try {
              torch::Tensor center = kf->getCenter();

              // Access center data (x, y, z)
              auto center_cpu = center.cpu();
              float* center_data = center_cpu.data_ptr<float>();

              // Use X and Z for top-down view (bird's eye view)
              float x = center_data[0];
              float z = center_data[2];

              positions.push_back({x, z});

              // Collect additional data for frustum visualization
              KeyframeData kf_data;
              kf_data.x = x;
              kf_data.z = z;
              kf_data.rotation = kf->getRotationMatrixf();
              kf_data.fov_x = kf->FoVx_;
              kf_data.zfar = kf->zfar_;
              keyframe_data.push_back(kf_data);
            } catch (const std::exception& e) {
              // Skip keyframes with errors
              continue;
            }
          }
        }

        // Draw trajectory if we have positions
        if (!positions.empty()) {
          // Find min/max for scaling
          float min_x = positions[0].first;
          float max_x = positions[0].first;
          float min_z = positions[0].second;
          float max_z = positions[0].second;

          for (const auto& pos : positions) {
            min_x = std::min(min_x, pos.first);
            max_x = std::max(max_x, pos.first);
            min_z = std::min(min_z, pos.second);
            max_z = std::max(max_z, pos.second);
          }

          // Calculate scale and center
          float range_x = max_x - min_x;
          float range_z = max_z - min_z;
          float max_range = std::max(range_x, range_z);

          // Avoid division by zero
          if (max_range < 0.001f) {
            max_range = 1.0f;
          }

          float scale = (window_size_ - 2 * margin_) / max_range;
          float center_x = (min_x + max_x) / 2.0f;
          float center_z = (min_z + max_z) / 2.0f;

          // Draw trajectory lines
          for (size_t i = 1; i < positions.size(); ++i) {
            // Transform to image coordinates
            int x1 = static_cast<int>(
                (positions[i - 1].first - center_x) * scale + window_size_ / 2);
            int y1 =
                static_cast<int>(window_size_ / 2 -
                                 (positions[i - 1].second - center_z) * scale);
            int x2 = static_cast<int>((positions[i].first - center_x) * scale +
                                      window_size_ / 2);
            int y2 = static_cast<int>(window_size_ / 2 -
                                      (positions[i].second - center_z) * scale);

            cv::line(display, cv::Point(x1, y1), cv::Point(x2, y2),
                     cv::Scalar(0, 0, 0), 2);
          }

          // Draw frustums for each keyframe
          for (const auto& kf_data : keyframe_data) {
            // Camera coordinate system: X right, Y down, Z forward
            // Extract forward direction from rotation matrix (third row for
            // world-to-camera)
            Eigen::Vector3f forward = kf_data.rotation.row(2);

            // Project forward direction to X-Z plane (top-down view)
            float forward_x = forward(0);
            float forward_z = forward(2);
            float forward_norm =
                std::sqrt(forward_x * forward_x + forward_z * forward_z);

            if (forward_norm > 0.001f) {
              forward_x /= forward_norm;
              forward_z /= forward_norm;

              // Calculate frustum corners at zfar distance
              float half_fov = kf_data.fov_x / 2.0f;

              // Center point of frustum far plane
              float far_center_x = kf_data.x + forward_x * kf_data.zfar;
              float far_center_z = kf_data.z + forward_z * kf_data.zfar;

              // Calculate left and right vectors (perpendicular to forward in
              // X-Z plane)
              float right_x = forward_z;  // Perpendicular in X-Z plane
              float right_z = -forward_x;

              // Width of frustum at far plane
              float far_width = 2.0f * kf_data.zfar * std::tan(half_fov);

              // Left and right corners of frustum
              float left_x = far_center_x - right_x * far_width / 2.0f;
              float left_z = far_center_z - right_z * far_width / 2.0f;
              float right_x_pos = far_center_x + right_x * far_width / 2.0f;
              float right_z_pos = far_center_z + right_z * far_width / 2.0f;

              // Transform to screen coordinates
              int cam_screen_x = static_cast<int>(
                  (kf_data.x - center_x) * scale + window_size_ / 2);
              int cam_screen_y = static_cast<int>(
                  window_size_ / 2 - (kf_data.z - center_z) * scale);

              int left_screen_x = static_cast<int>((left_x - center_x) * scale +
                                                   window_size_ / 2);
              int left_screen_y = static_cast<int>(window_size_ / 2 -
                                                   (left_z - center_z) * scale);

              int right_screen_x = static_cast<int>(
                  (right_x_pos - center_x) * scale + window_size_ / 2);
              int right_screen_y = static_cast<int>(
                  window_size_ / 2 - (right_z_pos - center_z) * scale);

              // Draw frustum lines in green
              cv::Scalar frustum_color(0, 150, 0);  // Green
              int line_thickness = 1;

              // Left edge
              cv::line(display, cv::Point(cam_screen_x, cam_screen_y),
                       cv::Point(left_screen_x, left_screen_y), frustum_color,
                       line_thickness);

              // Right edge
              cv::line(display, cv::Point(cam_screen_x, cam_screen_y),
                       cv::Point(right_screen_x, right_screen_y), frustum_color,
                       line_thickness);

              // Far edge (connecting left and right)
              cv::line(display, cv::Point(left_screen_x, left_screen_y),
                       cv::Point(right_screen_x, right_screen_y), frustum_color,
                       line_thickness);
            }
          }

          // Build set for fast lookup of recently selected keyframes
          std::unordered_set<int> recent_selected_set;
          {
            std::lock_guard<std::mutex> lock(selection_mutex_);
            recent_selected_set.insert(recent_selected_keyframe_ids_.begin(),
                                       recent_selected_keyframe_ids_.end());
          }

          // Draw keyframe points with color-coding (on top of frustums)
          auto keyframe_it = keyframes_map.rbegin();
          for (size_t i = 0; i < positions.size(); ++i, ++keyframe_it) {
            const auto& pos = positions[i];
            int x = static_cast<int>((pos.first - center_x) * scale +
                                     window_size_ / 2);
            int y = static_cast<int>(window_size_ / 2 -
                                     (pos.second - center_z) * scale);

            int kf_id = static_cast<int>(keyframe_it->second->fid_);

            // Determine color based on selection status
            cv::Scalar color;
            if (recent_selected_set.count(kf_id) > 0) {
              // Recently selected: Orange
              color = cv::Scalar(0, 165, 255);
            } else {
              // Default: Blue
              color = cv::Scalar(255, 0, 0);
            }

            cv::circle(display, cv::Point(x, y), 4, color, -1);
          }

          // Draw info text
          std::string info = "Keyframes: " + std::to_string(positions.size());
          cv::putText(display, info, cv::Point(10, 30),
                      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

          // Draw selection info
          if (!recent_selected_set.empty()) {
            std::string selection_info =
                "Last 100 selected: " +
                std::to_string(recent_selected_set.size()) + " unique";
            cv::putText(display, selection_info, cv::Point(10, 60),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
          }
        }
      }
    }

    cv::imshow(window_name, display);

    // Wait for 1 second (1 FPS) and handle window events
    cv::waitKey(1000);
  }

  cv::destroyWindow(window_name);
}

bool TrajectoryViewer::isStopped() {
  std::unique_lock<std::mutex> lock(mutex_status_);
  return stopped_;
}

void TrajectoryViewer::signalStop(const bool going_to_stop) {
  std::unique_lock<std::mutex> lock(mutex_status_);
  stopped_ = going_to_stop;
}

void TrajectoryViewer::recordKeyframeSelection(int keyframe_id) {
  std::lock_guard<std::mutex> lock(selection_mutex_);
  recent_selected_keyframe_ids_.push_back(keyframe_id);
  if (recent_selected_keyframe_ids_.size() > max_selection_history_) {
    recent_selected_keyframe_ids_.pop_front();
  }
}

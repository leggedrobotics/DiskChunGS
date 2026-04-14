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

#include "scene/gaussian_scene.h"

#include <iostream>

GaussianScene::GaussianScene(GaussianModelParams& args, int load_iteration) {
  if (load_iteration) {
    loaded_iter_ = load_iteration;
    std::cout << "Loading trained model at iteration " << load_iteration
              << std::endl;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Camera management
// ─────────────────────────────────────────────────────────────────────────────

void GaussianScene::addCamera(Camera& camera) {
  cameras_.emplace(camera.camera_id_, camera);
}

Camera& GaussianScene::getCamera(camera_id_t camera_id) {
  return cameras_[camera_id];
}

// ─────────────────────────────────────────────────────────────────────────────
// Keyframe management
// ─────────────────────────────────────────────────────────────────────────────

void GaussianScene::addKeyframe(std::shared_ptr<GaussianKeyframe> keyframe) {
  std::unique_lock<std::mutex> lock(mutex_kfs_);
  keyframes_.emplace(keyframe->fid_, keyframe);
}

std::shared_ptr<GaussianKeyframe> GaussianScene::getKeyframe(std::size_t fid) {
  std::unique_lock<std::mutex> lock(mutex_kfs_);
  auto it = keyframes_.find(fid);
  return (it != keyframes_.end()) ? it->second : nullptr;
}

std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>&
GaussianScene::keyframes() {
  return keyframes_;
}

std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>
GaussianScene::getAllKeyframes() {
  std::unique_lock<std::mutex> lock(mutex_kfs_);
  return keyframes_;
}

// ─────────────────────────────────────────────────────────────────────────────
// Transformations and normalization
// ─────────────────────────────────────────────────────────────────────────────

void GaussianScene::applyScaledTransformation(const float scale,
                                              const Sophus::SE3f transform) {
  for (auto& [fid, keyframe] : keyframes_) {
    Sophus::SE3f Twc = keyframe->getPosef().inverse();
    Twc.translation() *= scale;

    Sophus::SE3f Tcy = (transform * Twc).inverse();
    keyframe->setPose(Tcy.unit_quaternion().cast<double>(),
                      Tcy.translation().cast<double>());
    keyframe->computeTransformTensors();
  }
}

std::tuple<Eigen::Vector3f, float> GaussianScene::getNerfppNorm() {
  auto kfs = getAllKeyframes();
  const std::size_t n_cams = kfs.size();

  // Collect camera centers from world-to-camera transforms
  std::vector<Eigen::Vector3f> cam_centers;
  cam_centers.reserve(n_cams);
  for (const auto& [fid, keyframe] : kfs) {
    Eigen::Matrix4f C2W = keyframe->getWorld2View2().inverse();
    cam_centers.emplace_back(C2W.block<3, 1>(0, 3));
  }

  // Compute centroid
  Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
  for (const auto& center : cam_centers) {
    centroid += center;
  }
  centroid /= static_cast<float>(n_cams);

  // Find maximum distance from centroid (bounding radius with 10% margin)
  float max_dist = 0.0f;
  for (const auto& center : cam_centers) {
    max_dist = std::max(max_dist, (center - centroid).norm());
  }

  return {-centroid, max_dist * 1.1f};
}
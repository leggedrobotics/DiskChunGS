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

#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

#include "camera.h"
#include "geometry/point3d.h"
#include "model/gaussian_model.h"
#include "scene/gaussian_keyframe.h"
#include "gaussian_splatting/scene/gaussian_parameters.h"
#include "types.h"

/**
 * @brief Manages a collection of cameras and keyframes for
 *        Gaussian splatting reconstruction.
 *
 * This class serves as the central container for scene data, providing
 * thread-safe access to keyframes and methods for coordinate transformations.
 */
class GaussianScene {
 public:
  /**
   * @brief Constructs a GaussianScene.
   * @param args Model parameters configuration.
   * @param load_iteration Iteration to load from disk (0 = no loading).
   */
  GaussianScene(GaussianModelParams& args, int load_iteration = 0);

  // ─────────────────────────────────────────────────────────────────────────
  // Camera management
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * @brief Adds a camera to the scene.
   * @param camera Camera to add (stored by camera_id_).
   */
  void addCamera(Camera& camera);

  /**
   * @brief Retrieves a camera by ID.
   * @param camera_id The camera identifier.
   * @return Reference to the camera.
   */
  Camera& getCamera(camera_id_t camera_id);

  // ─────────────────────────────────────────────────────────────────────────
  // Keyframe management (thread-safe)
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * @brief Adds a keyframe to the scene (thread-safe).
   * @param keyframe Shared pointer to the keyframe.
   */
  void addKeyframe(std::shared_ptr<GaussianKeyframe> keyframe);

  /**
   * @brief Retrieves a keyframe by frame ID (thread-safe).
   * @param fid Frame identifier.
   * @return Shared pointer to keyframe, or nullptr if not found.
   */
  std::shared_ptr<GaussianKeyframe> getKeyframe(std::size_t fid);

  /**
   * @brief Returns a reference to the keyframes map (not thread-safe).
   * @warning Direct access bypasses mutex protection.
   */
  std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>& keyframes();

  /**
   * @brief Returns a copy of all keyframes (thread-safe).
   * @return Copy of the keyframes map.
   */
  std::map<std::size_t, std::shared_ptr<GaussianKeyframe>> getAllKeyframes();

  // ─────────────────────────────────────────────────────────────────────────
  // Transformations and normalization
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * @brief Applies a scaled SE3 transformation to all keyframes.
   * @param scale Scale factor applied to translations.
   * @param transform SE3 transformation to apply.
   */
  void applyScaledTransformation(
      float scale = 1.0f,
      Sophus::SE3f transform = Sophus::SE3f(Eigen::Matrix3f::Identity(),
                                            Eigen::Vector3f::Zero()));

  /**
   * @brief Computes NeRF++ style normalization from camera positions.
   *
   * Calculates the centroid of all camera centers and the radius of the
   * bounding sphere (with 10% margin).
   *
   * @return Tuple of (translation to center, bounding radius).
   */
  std::tuple<Eigen::Vector3f, float> getNerfppNorm();

  // ─────────────────────────────────────────────────────────────────────────
  // Public data members
  // ─────────────────────────────────────────────────────────────────────────

  float cameras_extent_ = 0.0f;  ///< Scene radius for NeRF normalization.
  int loaded_iter_ = 0;          ///< Iteration loaded from disk (0 = none).

  std::map<camera_id_t, Camera> cameras_;
  std::map<std::size_t, std::shared_ptr<GaussianKeyframe>> keyframes_;
  std::map<point3D_id_t, Point3D> cached_point_cloud_;

 protected:
  std::mutex mutex_kfs_;  ///< Mutex for thread-safe keyframe access.
};
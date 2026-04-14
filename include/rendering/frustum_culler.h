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

#include <Eigen/Dense>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"
#include "chunk_types.h"

class GaussianKeyframe;

/**
 * @brief Result of testing an AABB against a view frustum.
 */
enum FrustumTestResult { OUTSIDE = 0, INTERSECT = 1, INSIDE = 2 };

/**
 * @brief Performs frustum culling tests against axis-aligned bounding boxes.
 *
 * Extracts and normalizes the six frustum planes from a model-view-projection
 * matrix, then provides AABB intersection tests.
 */
class FrustumCuller {
 public:
  /**
   * @brief Constructs a FrustumCuller from a model-view-projection matrix.
   * @param MVP The combined model-view-projection matrix.
   */
  explicit FrustumCuller(const Eigen::Matrix4f& MVP);

  /**
   * @brief Tests an AABB against the frustum planes.
   * @param aabb The axis-aligned bounding box to test.
   * @return OUTSIDE if fully outside, INSIDE if fully inside, INTERSECT if
   *         partially visible.
   */
  FrustumTestResult testAABB(const AABB& aabb) const;

 private:
  Eigen::Vector4f planes_[6];  ///< Normalized frustum planes (left, right,
                               ///< bottom, top, near, far)
};

/**
 * @brief Performs hierarchical frustum culling on a grid of chunks.
 *
 * Uses octree-style subdivision to efficiently cull large regions, with
 * optional parallel processing for improved performance.
 *
 * @param view_projection_matrix Combined view-projection matrix for frustum
 *                               extraction.
 * @param camera_position World-space camera position for distance culling.
 * @param camera_chunk Chunk coordinate containing the camera.
 * @param search_radius Radius (in chunks) around camera_chunk to consider.
 * @param chunk_size World-space size of each chunk.
 * @param max_distance Maximum distance for chunk visibility.
 * @return Vector of visible chunk coordinates.
 */
std::vector<ChunkCoord> cullChunksHierarchical(
    const Eigen::Matrix4f& view_projection_matrix,
    const Eigen::Vector3f& camera_position,
    const ChunkCoord& camera_chunk,
    int search_radius,
    float chunk_size,
    float max_distance);

/**
 * @brief Cache for frustum culling results to avoid redundant computation.
 *
 * Stores visible chunks per keyframe and reuses results when the keyframe
 * pose hasn't changed significantly.
 */
class FrustumCullingCache {
 public:
  /**
   * @brief Cached visibility data for a single keyframe.
   */
  struct CacheEntry {
    Sophus::SE3d pose;                       ///< Pose when visibility was computed
    std::vector<ChunkCoord> visible_chunks;  ///< Cached visible chunks
  };

  /**
   * @brief Constructs a cache with the given chunk size for tolerance
   *        calculation.
   * @param chunk_size World-space chunk size.
   */
  explicit FrustumCullingCache(float chunk_size) : chunk_size_(chunk_size) {}

  /**
   * @brief Attempts to retrieve cached visibility results.
   * @param keyframe_id Unique identifier for the keyframe.
   * @param current_pose Current pose of the keyframe.
   * @param[out] out_chunks Output vector for cached chunks if found.
   * @return True if valid cached result was found and returned.
   */
  bool getCached(size_t keyframe_id,
                 const Sophus::SE3d& current_pose,
                 std::vector<ChunkCoord>& out_chunks);

  /**
   * @brief Updates the cache with new visibility results.
   * @param keyframe_id Unique identifier for the keyframe.
   * @param pose Pose at which visibility was computed.
   * @param visible_chunks The computed visible chunks.
   */
  void updateCache(size_t keyframe_id,
                   const Sophus::SE3d& pose,
                   const std::vector<ChunkCoord>& visible_chunks);

  /**
   * @brief Clears all cached entries.
   */
  void clearCache() { cache_.clear(); }

 private:
  float chunk_size_;
  std::unordered_map<size_t, CacheEntry> cache_;

  /**
   * @brief Checks if two poses are similar enough to reuse cached results.
   * @param a First pose.
   * @param b Second pose.
   * @return True if poses are within tolerance thresholds.
   */
  bool poseNearlyEqual(const Sophus::SE3d& a, const Sophus::SE3d& b) const;
};

/**
 * @brief Computes visible chunks for a keyframe with optional caching.
 *
 * Extracts camera parameters from the keyframe, performs hierarchical
 * frustum culling, and optionally caches results for reuse.
 *
 * @param keyframe The keyframe to compute visibility for.
 * @param chunk_size World-space chunk size.
 * @param cache Optional cache for result storage/retrieval.
 * @return Vector of visible chunk coordinates.
 */
std::vector<ChunkCoord> frustumCullChunks(
    std::shared_ptr<GaussianKeyframe> keyframe,
    float chunk_size,
    FrustumCullingCache* cache = nullptr);
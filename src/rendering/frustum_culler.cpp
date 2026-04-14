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

#include "rendering/frustum_culler.h"

#include <omp.h>
#include <torch/torch.h>

#include <cmath>
#include <mutex>

#include "scene/gaussian_keyframe.h"

// ============================================================================
// FrustumCuller
// ============================================================================

FrustumCuller::FrustumCuller(const Eigen::Matrix4f& MVP) {
  // Extract frustum planes from MVP matrix using Gribb-Hartmann method
  planes_[0] = MVP.row(3) + MVP.row(0);  // Left
  planes_[1] = MVP.row(3) - MVP.row(0);  // Right
  planes_[2] = MVP.row(3) + MVP.row(1);  // Bottom
  planes_[3] = MVP.row(3) - MVP.row(1);  // Top
  planes_[4] = MVP.row(3) + MVP.row(2);  // Near
  planes_[5] = MVP.row(3) - MVP.row(2);  // Far

  // Normalize planes for accurate distance calculations
  for (int i = 0; i < 6; ++i) {
    float length = planes_[i].head<3>().norm();
    if (length > 0) {
      planes_[i] /= length;
    }
  }
}

FrustumTestResult FrustumCuller::testAABB(const AABB& aabb) const {
  bool intersects = false;

  for (int i = 0; i < 6; ++i) {
    Eigen::Vector3f normal = planes_[i].head<3>();
    float d = planes_[i].w();

    // P-vertex: farthest point in direction of plane normal
    Eigen::Vector3f p_vertex = aabb.min;
    if (normal.x() > 0) p_vertex.x() = aabb.max.x();
    if (normal.y() > 0) p_vertex.y() = aabb.max.y();
    if (normal.z() > 0) p_vertex.z() = aabb.max.z();

    // N-vertex: nearest point in direction of plane normal
    Eigen::Vector3f n_vertex = aabb.max;
    if (normal.x() > 0) n_vertex.x() = aabb.min.x();
    if (normal.y() > 0) n_vertex.y() = aabb.min.y();
    if (normal.z() > 0) n_vertex.z() = aabb.min.z();

    float p_distance = normal.dot(p_vertex) + d;
    float n_distance = normal.dot(n_vertex) + d;

    // Both vertices behind plane: completely outside
    if (p_distance < 0 && n_distance < 0) {
      return FrustumTestResult::OUTSIDE;
    }

    // One vertex behind plane: intersecting
    if (p_distance < 0 || n_distance < 0) {
      intersects = true;
    }
  }

  return intersects ? FrustumTestResult::INTERSECT : FrustumTestResult::INSIDE;
}

// ============================================================================
// FrustumCullingCache
// ============================================================================

bool FrustumCullingCache::getCached(size_t keyframe_id,
                                    const Sophus::SE3d& current_pose,
                                    std::vector<ChunkCoord>& out_chunks) {
  auto cache_it = cache_.find(keyframe_id);
  if (cache_it != cache_.end() &&
      poseNearlyEqual(current_pose, cache_it->second.pose)) {
    out_chunks = cache_it->second.visible_chunks;
    return true;
  }
  return false;
}

void FrustumCullingCache::updateCache(
    size_t keyframe_id,
    const Sophus::SE3d& pose,
    const std::vector<ChunkCoord>& visible_chunks) {
  cache_[keyframe_id] = {pose, visible_chunks};
}

bool FrustumCullingCache::poseNearlyEqual(const Sophus::SE3d& a,
                                          const Sophus::SE3d& b) const {
  constexpr double kRotationTol = 0.05;  // ~3 degrees
  const double translation_tol = chunk_size_ * 0.05;

  return (a.translation() - b.translation()).norm() < translation_tol &&
         a.unit_quaternion().angularDistance(b.unit_quaternion()) <
             kRotationTol;
}

// ============================================================================
// Hierarchical Chunk Culling
// ============================================================================

std::vector<ChunkCoord> cullChunksHierarchical(
    const Eigen::Matrix4f& view_projection_matrix,
    const Eigen::Vector3f& camera_position,
    const ChunkCoord& camera_chunk,
    int search_radius,
    float chunk_size,
    float max_distance) {
  FrustumCuller culler(view_projection_matrix);

  std::vector<ChunkCoord> visible_chunks;
  visible_chunks.reserve(1000);

  std::mutex result_mutex;

  // Recursive octree-style culling
  std::function<void(ChunkCoord, ChunkCoord, int)> cullRegion =
      [&](ChunkCoord min_coord, ChunkCoord max_coord, int depth) {
        AABB region_aabb = getRegionAABB(min_coord, max_coord, chunk_size);
        FrustumTestResult result = culler.testAABB(region_aabb);

        if (result == OUTSIDE) {
          return;
        }

        int64_t dx = max_coord.x - min_coord.x + 1;
        int64_t dy = max_coord.y - min_coord.y + 1;
        int64_t dz = max_coord.z - min_coord.z + 1;
        int64_t volume = dx * dy * dz;

        // Base case: process individual chunks
        if (result == INSIDE || (dx <= 2 && dy <= 2 && dz <= 2) || depth > 4) {
          bool use_parallel = (volume > 64) && (depth <= 2);

          if (use_parallel) {
#pragma omp parallel
            {
              std::vector<ChunkCoord> thread_chunks;
              thread_chunks.reserve(64);

#pragma omp for collapse(3) schedule(dynamic, 8)
              for (int64_t x = min_coord.x; x <= max_coord.x; ++x) {
                for (int64_t y = min_coord.y; y <= max_coord.y; ++y) {
                  for (int64_t z = min_coord.z; z <= max_coord.z; ++z) {
                    ChunkCoord coord{x, y, z};
                    Eigen::Vector3f chunk_center =
                        getChunkCenter(coord, chunk_size);

                    if ((chunk_center - camera_position).norm() <=
                        max_distance) {
                      if (result == INSIDE) {
                        thread_chunks.push_back(coord);
                      } else {
                        AABB chunk_aabb = getChunkAABB(coord, chunk_size);
                        if (culler.testAABB(chunk_aabb) != OUTSIDE) {
                          thread_chunks.push_back(coord);
                        }
                      }
                    }
                  }
                }
              }

              if (!thread_chunks.empty()) {
                std::lock_guard<std::mutex> lock(result_mutex);
                visible_chunks.insert(visible_chunks.end(),
                                      thread_chunks.begin(),
                                      thread_chunks.end());
              }
            }
          } else {
            std::vector<ChunkCoord> local_chunks;

            for (int64_t x = min_coord.x; x <= max_coord.x; ++x) {
              for (int64_t y = min_coord.y; y <= max_coord.y; ++y) {
                for (int64_t z = min_coord.z; z <= max_coord.z; ++z) {
                  ChunkCoord coord{x, y, z};
                  Eigen::Vector3f chunk_center =
                      getChunkCenter(coord, chunk_size);

                  if ((chunk_center - camera_position).norm() <= max_distance) {
                    if (result == INSIDE) {
                      local_chunks.push_back(coord);
                    } else {
                      AABB chunk_aabb = getChunkAABB(coord, chunk_size);
                      if (culler.testAABB(chunk_aabb) != OUTSIDE) {
                        local_chunks.push_back(coord);
                      }
                    }
                  }
                }
              }
            }

            if (!local_chunks.empty()) {
              std::lock_guard<std::mutex> lock(result_mutex);
              visible_chunks.insert(visible_chunks.end(), local_chunks.begin(),
                                    local_chunks.end());
            }
          }
          return;
        }

        // Subdivide into octants
        int64_t mid_x = (min_coord.x + max_coord.x) / 2;
        int64_t mid_y = (min_coord.y + max_coord.y) / 2;
        int64_t mid_z = (min_coord.z + max_coord.z) / 2;

        auto recurse = [&](ChunkCoord lo, ChunkCoord hi) {
          cullRegion(lo, hi, depth + 1);
        };

        if (depth <= 1 && volume > 512) {
#pragma omp parallel sections
          {
#pragma omp section
            recurse({min_coord.x, min_coord.y, min_coord.z},
                    {mid_x, mid_y, mid_z});
#pragma omp section
            recurse({mid_x + 1, min_coord.y, min_coord.z},
                    {max_coord.x, mid_y, mid_z});
#pragma omp section
            recurse({min_coord.x, mid_y + 1, min_coord.z},
                    {mid_x, max_coord.y, mid_z});
#pragma omp section
            recurse({mid_x + 1, mid_y + 1, min_coord.z},
                    {max_coord.x, max_coord.y, mid_z});
#pragma omp section
            recurse({min_coord.x, min_coord.y, mid_z + 1},
                    {mid_x, mid_y, max_coord.z});
#pragma omp section
            recurse({mid_x + 1, min_coord.y, mid_z + 1},
                    {max_coord.x, mid_y, max_coord.z});
#pragma omp section
            recurse({min_coord.x, mid_y + 1, mid_z + 1},
                    {mid_x, max_coord.y, max_coord.z});
#pragma omp section
            recurse({mid_x + 1, mid_y + 1, mid_z + 1},
                    {max_coord.x, max_coord.y, max_coord.z});
          }
        } else {
          recurse({min_coord.x, min_coord.y, min_coord.z},
                  {mid_x, mid_y, mid_z});
          recurse({mid_x + 1, min_coord.y, min_coord.z},
                  {max_coord.x, mid_y, mid_z});
          recurse({min_coord.x, mid_y + 1, min_coord.z},
                  {mid_x, max_coord.y, mid_z});
          recurse({mid_x + 1, mid_y + 1, min_coord.z},
                  {max_coord.x, max_coord.y, mid_z});
          recurse({min_coord.x, min_coord.y, mid_z + 1},
                  {mid_x, mid_y, max_coord.z});
          recurse({mid_x + 1, min_coord.y, mid_z + 1},
                  {max_coord.x, mid_y, max_coord.z});
          recurse({min_coord.x, mid_y + 1, mid_z + 1},
                  {mid_x, max_coord.y, max_coord.z});
          recurse({mid_x + 1, mid_y + 1, mid_z + 1},
                  {max_coord.x, max_coord.y, max_coord.z});
        }
      };

  ChunkCoord min_coord{camera_chunk.x - search_radius,
                       camera_chunk.y - search_radius,
                       camera_chunk.z - search_radius};
  ChunkCoord max_coord{camera_chunk.x + search_radius,
                       camera_chunk.y + search_radius,
                       camera_chunk.z + search_radius};

  cullRegion(min_coord, max_coord, 0);

  return visible_chunks;
}

// ============================================================================
// Keyframe Frustum Culling
// ============================================================================

std::vector<ChunkCoord> frustumCullChunks(
    std::shared_ptr<GaussianKeyframe> keyframe,
    float chunk_size,
    FrustumCullingCache* cache) {
  if (!keyframe) {
    return {};
  }

  std::size_t keyframe_id = keyframe->fid_;
  Sophus::SE3d current_pose = keyframe->getPose();

  // Check cache
  if (cache) {
    std::vector<ChunkCoord> cached_chunks;
    if (cache->getCached(keyframe_id, current_pose, cached_chunks)) {
      return cached_chunks;
    }
  }

  // Build view-projection matrix
  Eigen::Matrix4f view_matrix =
      keyframe->getWorld2View2(keyframe->trans_, keyframe->scale_);

  torch::Tensor tensor_matrix = keyframe->projection_matrix_.cpu().contiguous();
  float* data_ptr = tensor_matrix.data_ptr<float>();
  Eigen::Matrix4f proj_matrix = Eigen::Map<Eigen::Matrix4f>(data_ptr);
  Eigen::Matrix4f vp_matrix = proj_matrix * view_matrix;

  // Camera position and chunk
  Sophus::SE3d Twc = current_pose.inverse();
  Eigen::Vector3f camera_position = Twc.translation().cast<float>();
  ChunkCoord camera_chunk = getChunkCoord(camera_position, chunk_size);

  // Culling parameters
  int search_radius = static_cast<int>(std::ceil(keyframe->zfar_ / chunk_size *
                                                 std::sqrt(3.0f))) +
                      2;
  float max_distance = keyframe->zfar_ + chunk_size * 1.732f;

  std::vector<ChunkCoord> visible_chunks =
      cullChunksHierarchical(vp_matrix, camera_position, camera_chunk,
                             search_radius, chunk_size, max_distance);

  // Update cache
  if (cache) {
    cache->updateCache(keyframe_id, current_pose, visible_chunks);
  }

  return visible_chunks;
}
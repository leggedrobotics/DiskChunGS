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

#include <deque>
#include <map>
#include <memory>
#include <random>

#include "scene/gaussian_keyframe.h"
#include "scene/gaussian_scene.h"

/**
 * @brief Manages keyframe selection for Gaussian Splatting optimization.
 *
 * This class implements a chunk-based keyframe selection strategy that:
 * - Groups keyframes by spatial chunks for locality-aware selection
 * - Uses loss-based weighting to prioritize high-error keyframes
 * - Manages GPU memory by maintaining a bounded queue of loaded keyframes
 * - Tracks usage statistics for balanced training coverage
 */
class KeyframeSelection {
 public:
  /**
   * @brief Constructs a keyframe selection manager.
   * @param scene Shared pointer to the Gaussian scene containing keyframes.
   * @param chunk_size Spatial size of each chunk for grouping keyframes.
   * @param auto_distribute Divisor for selecting top-k high-loss keyframes
   *                        (selects top 1/auto_distribute fraction).
   * @param loss_map Optional pointer to map of keyframe IDs to loss values.
   * @param used_times_map Optional pointer to map tracking keyframe usage
   * counts.
   */
  KeyframeSelection(std::shared_ptr<GaussianScene> scene,
                    float chunk_size = 200.0f,
                    int auto_distribute = 4,
                    const std::map<std::size_t, float>* loss_map = nullptr,
                    std::map<std::size_t, int>* used_times_map = nullptr);

  /**
   * @brief Selects the next keyframe for training based on the current chunk.
   *
   * Selection is weighted by remaining usage allowance and loss values.
   * Automatically manages GPU memory by loading/unloading keyframes as needed.
   *
   * @return Shared pointer to the selected keyframe, or nullptr if unavailable.
   */
  std::shared_ptr<GaussianKeyframe> getNextKeyframe();

  /**
   * @brief Updates the chunk-to-keyframe mapping for a given keyframe.
   * @param keyframe The keyframe to update mapping for.
   * @param is_new_keyframe If true, sets this as the latest keyframe and adds
   *                        it to its chunk. If false, removes old associations
   *                        before re-adding (for pose updates).
   */
  void updateChunkKeyframeMapping(std::shared_ptr<GaussianKeyframe> keyframe,
                                  bool is_new_keyframe = false);

  /**
   * @brief Returns the current size of the GPU keyframe queue.
   * @return Number of keyframes currently in the GPU queue.
   */
  int getQueueSize() const;

 private:
  std::shared_ptr<GaussianScene> scene_;
  float chunk_size_;
  int auto_distribute_;

  const std::map<std::size_t, float>* loss_map_;
  std::map<std::size_t, int>* used_times_map_;

  std::shared_ptr<GaussianKeyframe> latest_keyframe_;

  std::mt19937 rng_;

  std::map<int64_t, std::vector<std::shared_ptr<GaussianKeyframe>>>
      chunk_to_keyframes_;

  std::deque<std::shared_ptr<GaussianKeyframe>> gpu_queue_;
  size_t max_gpu_keyframes_ = 400;

  /** @brief Converts a 3D torch tensor to an Eigen vector. */
  Eigen::Vector3f tensorToEigen(const torch::Tensor& tensor) const;

  /** @brief Increases the remaining usage allowance for a keyframe. */
  void increaseKeyframeTimesOfUse(std::shared_ptr<GaussianKeyframe> keyframe,
                                  int additional_uses);
};
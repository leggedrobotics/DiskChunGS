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

#include "scene/keyframe_selection.h"

#include <algorithm>
#include <iostream>

KeyframeSelection::KeyframeSelection(
    std::shared_ptr<GaussianScene> scene,
    float chunk_size,
    int auto_distribute,
    const std::map<std::size_t, float>* loss_map,
    std::map<std::size_t, int>* used_times_map)
    : scene_(scene),
      chunk_size_(chunk_size),
      auto_distribute_(auto_distribute),
      loss_map_(loss_map),
      used_times_map_(used_times_map),
      rng_(std::random_device{}()) {}

std::shared_ptr<GaussianKeyframe> KeyframeSelection::getNextKeyframe() {
  if (!latest_keyframe_) {
    std::cout << "No latest keyframe available." << std::endl;
    return nullptr;
  }

  torch::Tensor latest_center = latest_keyframe_->getCenter();
  Eigen::Vector3f latest_position = tensorToEigen(latest_center);

  ChunkCoord chunk_coord = getChunkCoord(latest_position, chunk_size_);
  int64_t chunk_id = encodeChunkCoord(chunk_coord);

  auto it = chunk_to_keyframes_.find(chunk_id);
  if (it == chunk_to_keyframes_.end() || it->second.empty()) {
    std::cout << "Keyframe has moved to new chunk " << chunk_id
              << ", updating mapping..." << std::endl;

    // Keyframe's position has changed due to pose optimization - update its
    // chunk mapping
    updateChunkKeyframeMapping(latest_keyframe_, false);

    // Retry lookup after updating
    it = chunk_to_keyframes_.find(chunk_id);
    if (it == chunk_to_keyframes_.end() || it->second.empty()) {
      std::cerr << "ERROR: Still no keyframes in chunk " << chunk_id
                << " after updating mapping!" << std::endl;
      return nullptr;
    }
  }

  const auto& candidates = it->second;
  std::shared_ptr<GaussianKeyframe> selected_keyframe = nullptr;

  // Apply loss and usage-based selection logic
  std::vector<std::shared_ptr<GaussianKeyframe>> available_candidates;

  // First, check if any candidate has remaining times of use > 0
  for (const auto& candidate : candidates) {
    if (candidate->remaining_times_of_use_ > 0) {
      available_candidates.push_back(candidate);
    }
  }

  // If no candidates have remaining uses, apply loss-based distribution
  if (available_candidates.empty()) {
    // Basic increase for all candidates
    for (const auto& candidate : candidates) {
      increaseKeyframeTimesOfUse(candidate, 1);
      if (candidate->remaining_times_of_use_ > 0) {
        available_candidates.push_back(candidate);
      }
    }

    // Apply loss-based auto-distribution if loss_map is available
    if (loss_map_ && !loss_map_->empty()) {
      std::vector<std::pair<std::size_t, float>> loss_vec;

      // Collect loss values for candidates in this chunk
      for (const auto& candidate : candidates) {
        auto loss_it = loss_map_->find(candidate->fid_);
        if (loss_it != loss_map_->end()) {
          loss_vec.push_back({candidate->fid_, loss_it->second});
        }
      }

      if (!loss_vec.empty()) {
        // Select top (1/auto_distribute_) fraction of keyframes with highest
        // loss
        int k =
            std::max(1, static_cast<int>(loss_vec.size() / auto_distribute_));

        // Partial sort to get top-k highest loss keyframes
        std::nth_element(loss_vec.begin(), loss_vec.begin() + k, loss_vec.end(),
                         [](const std::pair<std::size_t, float>& a,
                            const std::pair<std::size_t, float>& b) {
                           return a.second > b.second;
                         });

        // Give additional uses to high-loss keyframes
        for (int i = 0; i < k; ++i) {
          auto scene_kf_it = scene_->keyframes().find(loss_vec[i].first);
          if (scene_kf_it != scene_->keyframes().end()) {
            for (const auto& candidate : candidates) {
              if (candidate->fid_ == loss_vec[i].first) {
                increaseKeyframeTimesOfUse(candidate, 1);
                break;
              }
            }
          }
        }

        // Refresh available candidates after loss-based distribution
        available_candidates.clear();
        for (const auto& candidate : candidates) {
          if (candidate->remaining_times_of_use_ > 0) {
            available_candidates.push_back(candidate);
          }
        }
      }
    }
  }

  // Select from available candidates using weighted random selection
  if (!available_candidates.empty()) {
    // Weight selection by remaining uses (higher remaining uses = higher
    // probability)
    std::vector<float> weights;
    weights.reserve(available_candidates.size());

    for (const auto& candidate : available_candidates) {
      float weight = std::max(
          1.0f, static_cast<float>(candidate->remaining_times_of_use_));
      weights.push_back(weight);
    }

    std::discrete_distribution<> weighted_dist(weights.begin(), weights.end());
    size_t selected_index = weighted_dist(rng_);
    selected_keyframe = available_candidates[selected_index];
  } else {
    // Last resort: random selection from all candidates
    std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
    size_t random_index = dist(rng_);
    selected_keyframe = candidates[random_index];
  }

  if (selected_keyframe) {
    // Update usage tracking
    if (used_times_map_) {
      auto used_times_it = used_times_map_->find(selected_keyframe->fid_);
      if (used_times_it == used_times_map_->end()) {
        used_times_map_->emplace(selected_keyframe->fid_, 1);
      } else {
        ++used_times_it->second;
      }
    }

    // Decrease remaining times of use
    if (selected_keyframe->remaining_times_of_use_ > 0) {
      --(selected_keyframe->remaining_times_of_use_);
    }

    // Ensure keyframe data is loaded to GPU
    if (!selected_keyframe->loaded_) {
      selected_keyframe->loadDataFromDisk();
    }

    // Remove keyframe if already in queue to avoid duplicates
    auto queue_it =
        std::find(gpu_queue_.begin(), gpu_queue_.end(), selected_keyframe);
    if (queue_it != gpu_queue_.end()) {
      gpu_queue_.erase(queue_it);
    }

    // Add to front (most recently used)
    gpu_queue_.push_front(selected_keyframe);

    // Evict oldest keyframes to stay within GPU memory budget
    while (gpu_queue_.size() > max_gpu_keyframes_) {
      std::shared_ptr<GaussianKeyframe> oldest = gpu_queue_.back();
      gpu_queue_.pop_back();

      if (oldest->loaded_ && oldest != selected_keyframe) {
        oldest->saveDataToDisk();
      }
    }
  }

  return selected_keyframe;
}

Eigen::Vector3f KeyframeSelection::tensorToEigen(
    const torch::Tensor& tensor) const {
  torch::Tensor cpu_tensor = tensor.cpu().contiguous();
  float* data_ptr = cpu_tensor.data_ptr<float>();
  return Eigen::Vector3f(data_ptr[0], data_ptr[1], data_ptr[2]);
}

void KeyframeSelection::increaseKeyframeTimesOfUse(
    std::shared_ptr<GaussianKeyframe> keyframe,
    int additional_uses) {
  if (!keyframe) return;
  keyframe->remaining_times_of_use_ += additional_uses;
}

void KeyframeSelection::updateChunkKeyframeMapping(
    std::shared_ptr<GaussianKeyframe> keyframe,
    bool is_new_keyframe) {
  if (!keyframe) return;

  if (is_new_keyframe) {
    latest_keyframe_ = keyframe;
  } else {
    // Remove old associations for updated keyframes
    for (auto& chunk_pair : chunk_to_keyframes_) {
      auto& keyframe_list = chunk_pair.second;
      keyframe_list.erase(
          std::remove(keyframe_list.begin(), keyframe_list.end(), keyframe),
          keyframe_list.end());
    }
  }

  // Add to chunk based on current position
  torch::Tensor center_tensor = keyframe->getCenter();
  Eigen::Vector3f position = tensorToEigen(center_tensor);
  ChunkCoord chunk_coord = getChunkCoord(position, chunk_size_);
  int64_t chunk_id = encodeChunkCoord(chunk_coord);
  chunk_to_keyframes_[chunk_id].push_back(keyframe);
}

int KeyframeSelection::getQueueSize() const { return gpu_queue_.size(); }
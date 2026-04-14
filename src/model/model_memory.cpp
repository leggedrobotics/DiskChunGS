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

#include "model/gaussian_model.h"

void GaussianModel::saveAndEvictChunks(const torch::Tensor& chunk_ids) {
  if (chunk_ids.size(0) == 0) return;

  // Mask of requested that are loaded
  torch::Tensor loaded_mask = torch::isin(chunk_ids, chunks_loaded_from_disk_);

  // IDs of requested that are loaded
  torch::Tensor chunks_to_save = chunk_ids.index({loaded_mask});

  // Save chunks that were previously loaded from disk
  if (chunks_to_save.size(0) > 0) {
    saveChunks(chunks_to_save);
  }

  // IDs of all gaussian's chunks in memory
  torch::Tensor spatial_chunks =
      std::get<0>(torch::_unique2(gaussian_chunk_ids_));

  // Mask of requested chunks that have gaussians in memory
  torch::Tensor has_gaussians_mask = torch::isin(chunk_ids, spatial_chunks);

  // Requested IDs of chunks that aren't loaded and have gaussians in memory
  torch::Tensor non_loaded_with_gaussians =
      chunk_ids.index({has_gaussians_mask & (~loaded_mask)});

  if (non_loaded_with_gaussians.size(0) > 0) {
    // Distinguish spillover vs new chunks

    // Spillover chunks: in memory but already saved on disk (will be discarded)
    torch::Tensor is_spillover_mask =
        torch::isin(non_loaded_with_gaussians, chunks_on_disk_);

    // New chunks: in memory but not yet on disk (need to be saved)
    torch::Tensor new_chunks =
        non_loaded_with_gaussians.index({~is_spillover_mask});

    // Save new chunks (spillover chunks are discarded without saving)
    if (new_chunks.size(0) > 0) {
      saveChunks(new_chunks); // Save the new chunks to disk
    }
  }

  // Remove all gaussians from evicted chunks. We have saved previously loaded
  // and new gaussians, only spillover gaussians remain. These are negligible
  torch::Tensor remove_mask = torch::isin(gaussian_chunk_ids_, chunk_ids);
  if (remove_mask.sum().item<int>() > 0) {
    prunePoints(remove_mask);
  }

  // Update loaded chunk tracking
  // Mask of loaded chunks not in the requested chunks to save
  torch::Tensor keep_loaded_mask =
      ~torch::isin(chunks_loaded_from_disk_, chunk_ids);

  // Remove requested chunk IDs that were in chunks_loaded_from_disk_
  chunks_loaded_from_disk_ = chunks_loaded_from_disk_.index({keep_loaded_mask});
}

size_t GaussianModel::getCurrentGPUMemoryUsage() const {
  if (torch::cuda::is_available()) {
    namespace c10Alloc = c10::cuda::CUDACachingAllocator;
    c10Alloc::DeviceStats mem_stats = c10Alloc::getDeviceStats(0);

    // Get current allocated bytes (this is what we want to track for chunks)
    c10Alloc::Stat alloc_bytes =
        mem_stats
            .allocated_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];
    return alloc_bytes.current;
  }
  return 0;
}

void GaussianModel::checkMemoryPressure() {
  int current_gaussians = getXYZ().size(0);
  if (current_gaussians <= max_gaussians_in_memory_) {
    return;  // No pressure, exit early
  }

  // Keep evicting until we reach memory goal or run out of chunks
  while (current_gaussians > max_gaussians_in_memory_) {
    torch::Tensor evictable_chunks =
        std::get<0>(torch::_unique2(gaussian_chunk_ids_));

    if (evictable_chunks.size(0) == 0) {
      std::cout << "[Memory] Warning: No evictable chunks found" << std::endl;
      break;
    }

    // Calculate how many gaussians to evict this iteration
    int64_t excess_gaussians = current_gaussians - max_gaussians_in_memory_;
    int64_t gaussians_to_evict =
        std::max(excess_gaussians, static_cast<int64_t>(100000));

    // Get LRU chunks that total at least gaussians_to_evict
    torch::Tensor lru_chunks =
        findLRUChunks(evictable_chunks, gaussians_to_evict);

    if (lru_chunks.size(0) == 0) {
      std::cout << "[Memory] Warning: No LRU chunks found to evict"
                << std::endl;
      break;
    }

    saveAndEvictChunks(lru_chunks);
    current_gaussians = getXYZ().size(0);
  }
}

torch::Tensor GaussianModel::findLRUChunks(
    const torch::Tensor& candidate_chunks,
    int64_t target_gaussian_count) {
  if (candidate_chunks.size(0) == 0 || target_gaussian_count <= 0) {
    return torch::empty(
        {0}, torch::TensorOptions().dtype(torch::kInt64).device(device_type_));
  }

  auto chunks_cpu = candidate_chunks.cpu();
  auto chunks_accessor = chunks_cpu.accessor<int64_t, 1>();

  // Gather chunk metadata: (chunk_id, access_time, gaussian_count)
  std::vector<std::tuple<int64_t, float, int64_t>> chunk_data;
  for (int64_t i = 0; i < chunks_cpu.size(0); i++) {
    int64_t chunk_id = chunks_accessor[i];
    float access_time = chunk_access_times_.count(chunk_id)
                            ? chunk_access_times_[chunk_id]
                            : 0.0f;
    torch::Tensor chunk_mask = (gaussian_chunk_ids_ == chunk_id);
    int64_t gaussian_count = chunk_mask.sum().item<int64_t>();
    chunk_data.emplace_back(chunk_id, access_time, gaussian_count);
  }

  // Sort by access time (oldest first for LRU eviction)
  std::sort(chunk_data.begin(), chunk_data.end(),
            [](const auto& a, const auto& b) {
              return std::get<1>(a) < std::get<1>(b);
            });

  // Select oldest chunks until target gaussian count is reached
  std::vector<int64_t> selected_chunks;
  int64_t accumulated_gaussians = 0;
  for (const auto& [chunk_id, access_time, gaussian_count] : chunk_data) {
    selected_chunks.push_back(chunk_id);
    accumulated_gaussians += gaussian_count;
    if (accumulated_gaussians >= target_gaussian_count) {
      break;
    }
  }

  // Convert to tensor
  torch::Tensor result = torch::empty(
      {static_cast<int64_t>(selected_chunks.size())},
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  auto result_accessor = result.accessor<int64_t, 1>();
  for (size_t i = 0; i < selected_chunks.size(); i++) {
    result_accessor[i] = selected_chunks[i];
  }

  return result.to(device_type_);
}

void GaussianModel::updateChunkAccess(const torch::Tensor& accessed_chunk_ids) {
  if (accessed_chunk_ids.size(0) == 0) return;

  float current_time = std::chrono::duration<float>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();

  auto chunk_ids_cpu = accessed_chunk_ids.cpu();
  auto chunk_ids_accessor = chunk_ids_cpu.accessor<int64_t, 1>();

  for (int64_t i = 0; i < chunk_ids_cpu.size(0); i++) {
    int64_t chunk_id = chunk_ids_accessor[i];
    chunk_access_times_[chunk_id] = current_time;
  }
}
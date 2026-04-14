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
#include "rendering/gaussian_rasterizer.h"

// =============================================================================
// Binary Tensor Serialization
// =============================================================================

void GaussianModel::saveTensorBinary(const torch::Tensor& tensor,
                                     std::ofstream& file) {
  TensorHeader header = {};
  header.dims = tensor.dim();
  for (int i = 0; i < tensor.dim(); ++i) {
    header.sizes[i] = static_cast<uint32_t>(tensor.size(i));
  }
  header.dtype = static_cast<uint32_t>(tensor.scalar_type());
  header.data_size = tensor.nbytes();

  file.write(reinterpret_cast<const char*>(&header), sizeof(header));

  torch::Tensor cpu_tensor = tensor.is_cuda() ? tensor.cpu() : tensor;
  file.write(reinterpret_cast<const char*>(cpu_tensor.data_ptr()),
             header.data_size);
}

torch::Tensor GaussianModel::loadTensorBinary(std::ifstream& file) {
  TensorHeader header;
  file.read(reinterpret_cast<char*>(&header), sizeof(header));

  std::vector<int64_t> sizes(header.dims);
  for (uint32_t i = 0; i < header.dims; ++i) {
    sizes[i] = header.sizes[i];
  }

  auto options = torch::TensorOptions()
                     .dtype(static_cast<torch::ScalarType>(header.dtype))
                     .device(torch::kCPU);
  torch::Tensor tensor = torch::empty(sizes, options);

  file.read(reinterpret_cast<char*>(tensor.data_ptr()), header.data_size);

  return tensor.to(device_type_);
}

// =============================================================================
// Chunk Filename & ID Helpers
// =============================================================================

std::string GaussianModel::getChunkFilename(const ChunkCoord& coord) {
  auto x_str = (coord.x >= 0 ? "p" : "n") + std::to_string(std::abs(coord.x));
  auto y_str = (coord.y >= 0 ? "p" : "n") + std::to_string(std::abs(coord.y));
  auto z_str = (coord.z >= 0 ? "p" : "n") + std::to_string(std::abs(coord.z));
  return storage_base_path_ + "/" + x_str + "_" + y_str + "_" + z_str + ".bin";
}

void GaussianModel::updateChunkIDs() {
  gaussian_chunk_ids_ = computeChunkIds(getXYZ(), chunk_size_);
}

// =============================================================================
// Single-Chunk Disk I/O
// =============================================================================

void GaussianModel::saveSingleChunkToDisk(int64_t chunk_id,
                                          const ChunkData& chunk_data) {
  std::string chunk_filename = getChunkFilename(decodeChunkCoord(chunk_id));

  std::filesystem::path chunk_path(chunk_filename);
  std::filesystem::create_directories(chunk_path.parent_path());

  std::ofstream file(chunk_filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + chunk_filename);
  }

  try {
    uint32_t magic = 0x43484E4B;  // "CHNK"
    uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));

    file.write(reinterpret_cast<const char*>(&chunk_id), sizeof(chunk_id));
    uint32_t num_points = static_cast<uint32_t>(chunk_data.num_points);
    file.write(reinterpret_cast<const char*>(&num_points), sizeof(num_points));

    // Gaussian parameters (order must match loadSingleChunkFromDisk)
    saveTensorBinary(chunk_data.xyz, file);
    saveTensorBinary(chunk_data.features_dc, file);
    saveTensorBinary(chunk_data.features_rest, file);
    saveTensorBinary(chunk_data.scaling, file);
    saveTensorBinary(chunk_data.rotation, file);
    saveTensorBinary(chunk_data.opacity, file);
    saveTensorBinary(chunk_data.exist_since, file);
    saveTensorBinary(chunk_data.position_lrs, file);
    saveTensorBinary(chunk_data.gaussian_ids, file);

    // Optimizer states (one per parameter group)
    for (int group_idx = 0; group_idx < kNumParamGroups; ++group_idx) {
      file.write(
          reinterpret_cast<const char*>(&chunk_data.step_counts[group_idx]),
          sizeof(int64_t));

      if (chunk_data.exp_avg_states[group_idx].defined()) {
        saveTensorBinary(chunk_data.exp_avg_states[group_idx], file);
        saveTensorBinary(chunk_data.exp_avg_sq_states[group_idx], file);
      } else {
        torch::Tensor empty = torch::empty({0});
        saveTensorBinary(empty, file);
        saveTensorBinary(empty, file);
      }
    }

    file.close();
  } catch (const std::exception& e) {
    file.close();
    std::filesystem::remove(chunk_filename);
    throw std::runtime_error("Failed to save chunk " +
                             std::to_string(chunk_id) + ": " + e.what());
  }
}

std::optional<GaussianModel::ChunkData> GaussianModel::loadSingleChunkFromDisk(
    int64_t chunk_id) {
  std::string chunk_filename = getChunkFilename(decodeChunkCoord(chunk_id));

  if (!std::filesystem::exists(chunk_filename)) {
    throw std::runtime_error("Chunk file does not exist: " + chunk_filename);
  }

  std::ifstream file(chunk_filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for reading: " + chunk_filename);
  }

  // Validate magic number and version
  uint32_t magic, version;
  file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  file.read(reinterpret_cast<char*>(&version), sizeof(version));
  if (magic != 0x43484E4B) {
    throw std::runtime_error("Invalid chunk file format: " + chunk_filename);
  }

  int64_t stored_chunk_id;
  uint32_t stored_num_points;
  file.read(reinterpret_cast<char*>(&stored_chunk_id), sizeof(stored_chunk_id));
  file.read(reinterpret_cast<char*>(&stored_num_points),
            sizeof(stored_num_points));
  if (stored_chunk_id != chunk_id) {
    throw std::runtime_error("Chunk ID mismatch in file: " + chunk_filename);
  }

  ChunkData data;
  try {
    // Gaussian parameters (order must match saveSingleChunkToDisk)
    data.xyz = loadTensorBinary(file);
    data.features_dc = loadTensorBinary(file);
    data.features_rest = loadTensorBinary(file);
    data.scaling = loadTensorBinary(file);
    data.rotation = loadTensorBinary(file);
    data.opacity = loadTensorBinary(file);
    data.exist_since = loadTensorBinary(file);
    data.position_lrs = loadTensorBinary(file);
    data.gaussian_ids = loadTensorBinary(file);

    // Optimizer states
    data.exp_avg_states.resize(kNumParamGroups);
    data.exp_avg_sq_states.resize(kNumParamGroups);
    data.step_counts.resize(kNumParamGroups);
    for (int group_idx = 0; group_idx < kNumParamGroups; ++group_idx) {
      file.read(reinterpret_cast<char*>(&data.step_counts[group_idx]),
                sizeof(int64_t));
      data.exp_avg_states[group_idx] = loadTensorBinary(file);
      data.exp_avg_sq_states[group_idx] = loadTensorBinary(file);
    }

    data.num_points = data.xyz.size(0);
    data.chunk_id = chunk_id;
    file.close();

    if (data.num_points != static_cast<int>(stored_num_points)) {
      std::cerr << "Point count mismatch in chunk file: " << chunk_filename
                << std::endl;
      return std::nullopt;
    }

    return data;
  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to load chunk " +
                             std::to_string(chunk_id) + ": " + e.what());
  }
}

// =============================================================================
// Chunk Data Extraction & Assembly
// =============================================================================

GaussianModel::ChunkData GaussianModel::extractChunkData(
    const torch::Tensor& chunk_mask,
    int64_t chunk_id) {
  ChunkData data;

  data.xyz = xyz_.index({chunk_mask}).detach().clone();
  data.features_dc = features_dc_.index({chunk_mask}).detach().clone();
  data.features_rest = features_rest_.index({chunk_mask}).detach().clone();
  data.scaling = scaling_.index({chunk_mask}).detach().clone();
  data.rotation = rotation_.index({chunk_mask}).detach().clone();
  data.opacity = opacity_.index({chunk_mask}).detach().clone();
  data.exist_since = exist_since_iter_.index({chunk_mask}).detach().clone();
  data.position_lrs = position_lrs_.index({chunk_mask}).detach().clone();
  data.gaussian_ids = gaussian_ids_.index({chunk_mask}).detach().clone();
  data.num_points = data.xyz.size(0);
  data.chunk_id = chunk_id;

  // Extract optimizer states
  data.exp_avg_states.resize(kNumParamGroups);
  data.exp_avg_sq_states.resize(kNumParamGroups);
  data.step_counts.resize(kNumParamGroups);

  auto& param_groups = optimizer_->param_groups();
  auto& state = optimizer_->state();

  for (int group_idx = 0; group_idx < kNumParamGroups; ++group_idx) {
    auto& param = param_groups[group_idx].params()[0];
    auto key = param.unsafeGetTensorImpl();

    if (state.find(key) == state.end()) {
      throw std::runtime_error("No optimizer state for param group " +
                               std::to_string(group_idx));
    }

    auto& param_state =
        static_cast<torch::optim::AdamParamState&>(*state[key]);
    data.exp_avg_states[group_idx] =
        param_state.exp_avg().index({chunk_mask}).detach().clone();
    data.exp_avg_sq_states[group_idx] =
        param_state.exp_avg_sq().index({chunk_mask}).detach().clone();
    data.step_counts[group_idx] = param_state.step();
  }

  return data;
}

void GaussianModel::appendLoadedChunks(
    const std::vector<ChunkData>& chunks_data,
    const std::vector<int64_t>& chunk_ids) {
  torch::NoGradGuard no_grad;
  if (chunks_data.empty()) return;

  // Collect per-chunk tensors for batch concatenation
  std::vector<torch::Tensor> all_xyz, all_features_dc, all_features_rest;
  std::vector<torch::Tensor> all_scaling, all_rotation, all_opacity;
  std::vector<torch::Tensor> all_exist_since, all_chunk_ids, all_position_lrs,
      all_gaussian_ids;

  std::vector<std::vector<torch::Tensor>> all_exp_avg(kNumParamGroups),
      all_exp_avg_sq(kNumParamGroups);
  std::vector<int64_t> max_step_counts(kNumParamGroups, 0);

  for (const auto& chunk : chunks_data) {
    all_xyz.push_back(chunk.xyz);
    all_features_dc.push_back(chunk.features_dc);
    all_features_rest.push_back(chunk.features_rest);
    all_scaling.push_back(chunk.scaling);
    all_rotation.push_back(chunk.rotation);
    all_opacity.push_back(chunk.opacity);
    all_exist_since.push_back(chunk.exist_since);
    all_position_lrs.push_back(chunk.position_lrs);
    all_gaussian_ids.push_back(chunk.gaussian_ids);

    all_chunk_ids.push_back(torch::full(
        {chunk.num_points}, chunk.chunk_id,
        torch::TensorOptions().device(device_type_).dtype(torch::kInt64)));

    for (int g = 0; g < kNumParamGroups; ++g) {
      if (chunk.exp_avg_states[g].defined()) {
        all_exp_avg[g].push_back(chunk.exp_avg_states[g]);
        all_exp_avg_sq[g].push_back(chunk.exp_avg_sq_states[g]);
      }
      max_step_counts[g] =
          std::max(max_step_counts[g], chunk.step_counts[g]);
    }
  }

  // Batch concatenation
  torch::Tensor batch_xyz = torch::cat(all_xyz, 0);
  torch::Tensor batch_features_dc = torch::cat(all_features_dc, 0);
  torch::Tensor batch_features_rest = torch::cat(all_features_rest, 0);
  torch::Tensor batch_scaling = torch::cat(all_scaling, 0);
  torch::Tensor batch_rotation = torch::cat(all_rotation, 0);
  torch::Tensor batch_opacity = torch::cat(all_opacity, 0);
  torch::Tensor batch_exist_since = torch::cat(all_exist_since, 0);
  torch::Tensor batch_position_lrs = torch::cat(all_position_lrs, 0);
  torch::Tensor batch_chunk_ids = torch::cat(all_chunk_ids, 0);
  torch::Tensor batch_gaussian_ids = torch::cat(all_gaussian_ids, 0);

  std::vector<torch::Tensor> concat_exp_avg(kNumParamGroups),
      concat_exp_avg_sq(kNumParamGroups);
  for (int g = 0; g < kNumParamGroups; ++g) {
    if (!all_exp_avg[g].empty()) {
      concat_exp_avg[g] = torch::cat(all_exp_avg[g], 0);
      concat_exp_avg_sq[g] = torch::cat(all_exp_avg_sq[g], 0);
    }
  }

  densificationPostfix(batch_xyz, batch_features_dc, batch_features_rest,
                       batch_opacity, batch_scaling, batch_rotation,
                       batch_exist_since, batch_chunk_ids, batch_position_lrs,
                       batch_gaussian_ids, concat_exp_avg, concat_exp_avg_sq,
                       max_step_counts);
}

// =============================================================================
// Eviction Helpers
// =============================================================================

void GaussianModel::evictExcessChunks(const torch::Tensor& protected_chunk_ids,
                                      int64_t excess_gaussians) {
  torch::Tensor spatial_chunks =
      std::get<0>(torch::_unique2(gaussian_chunk_ids_));
  if (spatial_chunks.size(0) == 0) return;

  torch::Tensor evictable_mask =
      ~torch::isin(spatial_chunks, protected_chunk_ids);
  torch::Tensor evictable_chunks = spatial_chunks.index({evictable_mask});
  if (evictable_chunks.size(0) == 0) {
    throw std::runtime_error(
        "No evictable chunks available - all chunks are protected");
  }

  // Hysteresis buffer (5%) to reduce eviction frequency
  int64_t buffer = static_cast<int64_t>(max_gaussians_in_memory_ * 0.05f);
  int64_t target_eviction = excess_gaussians + buffer;

  torch::Tensor lru_chunks = findLRUChunks(evictable_chunks, target_eviction);
  if (lru_chunks.size(0) == 0) {
    throw std::runtime_error(
        "No evictable chunks found after excluding protected chunks");
  }

  saveAndEvictChunks(lru_chunks);
}

int64_t GaussianModel::countGaussiansToLoad(
    const torch::Tensor& chunks_ids_needing_load) {
  auto to_load_cpu = chunks_ids_needing_load.cpu();
  auto chunks_on_disk_cpu = chunks_on_disk_.cpu();
  auto counts_cpu = chunk_gaussian_counts_.cpu();

  auto to_load_acc = to_load_cpu.accessor<int64_t, 1>();
  auto disk_acc = chunks_on_disk_cpu.accessor<int64_t, 1>();
  auto counts_acc = counts_cpu.accessor<int64_t, 1>();

  int64_t total = 0;
  for (int64_t i = 0; i < to_load_acc.size(0); i++) {
    int64_t target_id = to_load_acc[i];
    for (int64_t j = 0; j < disk_acc.size(0); j++) {
      if (disk_acc[j] == target_id) {
        total += counts_acc[j];
        break;
      }
    }
  }
  return total;
}

// =============================================================================
// Chunk Loading
// =============================================================================

void GaussianModel::loadChunks(const torch::Tensor& chunk_id_requests) {
  torch::NoGradGuard no_grad;
  if (chunk_id_requests.size(0) == 0) return;

  // Determine which requested chunks are on disk but not yet loaded
  torch::Tensor on_disk_mask = torch::isin(chunk_id_requests, chunks_on_disk_);
  torch::Tensor not_loaded_mask =
      ~torch::isin(chunk_id_requests, chunks_loaded_from_disk_);
  torch::Tensor chunks_ids_needing_load =
      chunk_id_requests.index({on_disk_mask & not_loaded_mask});

  // All requested chunks are already loaded -- just check memory pressure
  if (chunks_ids_needing_load.size(0) == 0) {
    int64_t current_gaussians = xyz_.size(0);
    if (current_gaussians > max_gaussians_in_memory_) {
      int64_t excess = current_gaussians - max_gaussians_in_memory_;
      std::cout << "[Load] Over limit by " << excess << " gaussians (have "
                << current_gaussians << ", max " << max_gaussians_in_memory_
                << ")" << std::endl;
      evictExcessChunks(chunk_id_requests, excess);
      std::cout << "[Load] Evicted non-visible chunks, new count: "
                << xyz_.size(0) << std::endl;
    }
    return;
  }

  // Pre-emptive eviction: ensure enough room for incoming Gaussians
  int64_t incoming = countGaussiansToLoad(chunks_ids_needing_load);
  int64_t projected_total = xyz_.size(0) + incoming;
  if (projected_total > max_gaussians_in_memory_) {
    int64_t excess = projected_total - max_gaussians_in_memory_;
    evictExcessChunks(chunks_ids_needing_load, excess);
  }

  // Parallel load from disk
  auto chunks_cpu = chunks_ids_needing_load.cpu();
  auto accessor = chunks_cpu.accessor<int64_t, 1>();
  int num_chunks = chunks_cpu.size(0);

  std::vector<std::future<std::pair<int64_t, std::optional<ChunkData>>>>
      futures;
  for (int i = 0; i < num_chunks; ++i) {
    int64_t chunk_id = accessor[i];
    futures.push_back(std::async(std::launch::async, [this, chunk_id]() {
      return std::make_pair(chunk_id, loadSingleChunkFromDisk(chunk_id));
    }));
  }

  // Collect and append results
  std::vector<ChunkData> chunks_to_append;
  std::vector<int64_t> loaded_chunk_ids;
  for (auto& future : futures) {
    auto [chunk_id, chunk_data] = future.get();
    if (chunk_data.has_value()) {
      chunks_to_append.push_back(std::move(chunk_data.value()));
      loaded_chunk_ids.push_back(chunk_id);
    }
  }

  if (!chunks_to_append.empty()) {
    appendLoadedChunks(chunks_to_append, loaded_chunk_ids);
  }

  // Track loaded chunks (deduplicated)
  chunks_loaded_from_disk_ =
      torch::cat({chunks_loaded_from_disk_, chunks_ids_needing_load}, 0);
  chunks_loaded_from_disk_ =
      std::get<0>(torch::_unique2(chunks_loaded_from_disk_));
}

// =============================================================================
// Chunk Saving
// =============================================================================

void GaussianModel::saveChunks(const torch::Tensor& chunk_ids_to_save) {
  torch::NoGradGuard no_grad;
  if (chunk_ids_to_save.size(0) == 0) return;

  auto chunks_cpu = chunk_ids_to_save.cpu();
  auto accessor = chunks_cpu.accessor<int64_t, 1>();
  int num_chunks = chunks_cpu.size(0);

  // Extract chunk data on main thread (requires GPU tensors)
  std::vector<std::pair<int64_t, ChunkData>> prepared_chunks;
  std::unordered_map<int64_t, int64_t> chunk_id_to_count;
  for (int i = 0; i < num_chunks; ++i) {
    int64_t chunk_id = accessor[i];
    torch::Tensor chunk_mask = (gaussian_chunk_ids_ == chunk_id);
    ChunkData chunk_data = extractChunkData(chunk_mask, chunk_id);
    chunk_id_to_count[chunk_id] = chunk_data.num_points;
    prepared_chunks.emplace_back(chunk_id, std::move(chunk_data));
  }

  // Write to disk in parallel (pure CPU/I/O)
  std::vector<std::future<std::pair<int64_t, bool>>> futures;
  for (auto& [chunk_id, chunk_data] : prepared_chunks) {
    futures.push_back(std::async(
        std::launch::async,
        [this](int64_t id, ChunkData data) {
          try {
            saveSingleChunkToDisk(id, data);
            return std::make_pair(id, true);
          } catch (const std::exception& e) {
            std::cerr << "Failed to save chunk " << id << ": " << e.what()
                      << std::endl;
            return std::make_pair(id, false);
          }
        },
        chunk_id, std::move(chunk_data)));
  }

  std::vector<int64_t> successfully_saved;
  for (auto& future : futures) {
    auto [chunk_id, success] = future.get();
    if (success) {
      successfully_saved.push_back(chunk_id);
    }
  }

  // Update disk tracking metadata
  if (chunk_gaussian_counts_.size(0) != chunks_on_disk_.size(0)) {
    throw std::runtime_error(
        "chunk_gaussian_counts_ and chunks_on_disk_ size mismatch");
  }

  for (int64_t chunk_id : successfully_saved) {
    torch::Tensor chunk_id_tensor = torch::tensor(
        {chunk_id},
        torch::TensorOptions().device(device_type_).dtype(torch::kInt64));
    torch::Tensor gaussian_count_tensor = torch::tensor(
        {chunk_id_to_count[chunk_id]},
        torch::TensorOptions().device(device_type_).dtype(torch::kInt64));

    auto mask = torch::eq(chunks_on_disk_, chunk_id_tensor);
    bool found = torch::any(mask).item<bool>();

    if (found) {
      // Update existing entry
      auto indices = torch::where(mask)[0];
      if (indices.size(0) > 1) {
        throw std::runtime_error("Duplicate chunk ID in chunks_on_disk_");
      }
      chunk_gaussian_counts_[indices[0].item<int64_t>()] =
          chunk_id_to_count[chunk_id];
    } else {
      // Add new entry
      chunks_on_disk_ = torch::cat({chunks_on_disk_, chunk_id_tensor}, 0);
      chunk_gaussian_counts_ =
          torch::cat({chunk_gaussian_counts_, gaussian_count_tensor}, 0);
    }
  }
}

void GaussianModel::saveAllChunks() {
  std::cout << "\n=== STARTING SAVE OF ALL CHUNKS IN MEMORY ===" << std::endl;

  torch::Tensor spatial_chunks =
      std::get<0>(torch::_unique2(gaussian_chunk_ids_));

  if (spatial_chunks.size(0) == 0) {
    std::cout << "No spatial chunks to save" << std::endl;
    return;
  }

  // Categorize chunks: loaded from disk, spillover (on disk but not loaded),
  // or new
  torch::Tensor loaded_mask =
      torch::isin(spatial_chunks, chunks_loaded_from_disk_);
  torch::Tensor on_disk_mask = torch::isin(spatial_chunks, chunks_on_disk_);
  torch::Tensor spillover_mask = on_disk_mask & ~loaded_mask;
  torch::Tensor new_mask = ~on_disk_mask & ~loaded_mask;

  torch::Tensor loaded_chunks = spatial_chunks.index({loaded_mask});
  torch::Tensor new_chunks = spatial_chunks.index({new_mask});

  std::cout << "Loaded chunks: " << loaded_chunks.size(0) << std::endl;
  std::cout << "Spillover chunks: "
            << spatial_chunks.index({spillover_mask}).size(0) << " (skipping)"
            << std::endl;
  std::cout << "New chunks: " << new_chunks.size(0) << std::endl;

  if (loaded_chunks.size(0) > 0) saveChunks(loaded_chunks);
  if (new_chunks.size(0) > 0) saveChunks(new_chunks);
  // Spillover chunks are not saved (disk data is more complete)

  std::cout << "=== SAVE COMPLETE ===" << std::endl;
}

// =============================================================================
// Gaussian Counting & Filtering
// =============================================================================

int64_t GaussianModel::countAllGaussians() {
  int64_t in_memory = xyz_.size(0);

  // Add only unloaded disk chunks (loaded ones are already counted)
  torch::Tensor unloaded_mask =
      ~torch::isin(chunks_on_disk_, chunks_loaded_from_disk_);
  int64_t on_disk =
      torch::sum(chunk_gaussian_counts_.index({unloaded_mask})).item<int64_t>();

  return in_memory + on_disk;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GaussianModel::filterPointsByChunkDensity(const torch::Tensor& xyz,
                                          const torch::Tensor& colors,
                                          const torch::Tensor& scales,
                                          const torch::Tensor& opacities,
                                          int min_gaussians_per_chunk) {
  if (min_gaussians_per_chunk <= 1) {
    return std::make_tuple(xyz, colors, scales, opacities);
  }

  torch::Tensor chunk_ids = computeChunkIds(xyz, chunk_size_);

  auto [unique_chunks, inverse_indices, counts] =
      torch::_unique2(chunk_ids, /*sorted=*/false,
                      /*return_inverse=*/true, /*return_counts=*/true);

  torch::Tensor valid_chunk_mask = counts >= min_gaussians_per_chunk;
  torch::Tensor valid_chunk_ids = unique_chunks.index({valid_chunk_mask});

  if (valid_chunk_ids.size(0) == 0) {
    return std::make_tuple(torch::empty({0, 3}, xyz.options()),
                           torch::empty({0, 3}, colors.options()),
                           scales.defined()
                               ? torch::empty({0, 3}, scales.options())
                               : torch::Tensor(),
                           torch::empty({0, 1}, opacities.options()));
  }

  torch::Tensor point_mask = torch::isin(chunk_ids, valid_chunk_ids);

  return std::make_tuple(
      xyz.index({point_mask}), colors.index({point_mask}),
      scales.defined() ? scales.index({point_mask}) : torch::Tensor(),
      opacities.index({point_mask}));
}

// =============================================================================
// Chunk Redistribution (post loop-closure)
// =============================================================================

void GaussianModel::handleBatchChunkRedistribution(
    const torch::Tensor& processed_chunk_ids) {
  torch::NoGradGuard no_grad;
  if (processed_chunk_ids.size(0) == 0) return;

  std::cout << "[Batch Redistribution] Processing "
            << processed_chunk_ids.size(0) << " chunks" << std::endl;

  // Find Gaussians belonging to processed chunks
  torch::Tensor chunk_mask =
      torch::isin(gaussian_chunk_ids_, processed_chunk_ids);
  torch::Tensor processed_indices = torch::where(chunk_mask)[0];

  if (processed_indices.size(0) == 0) {
    std::cout << "[Batch Redistribution] No gaussians found in processed chunks"
              << std::endl;
    return;
  }

  std::cout << "[Batch Redistribution] Found " << processed_indices.size(0)
            << " gaussians across all processed chunks" << std::endl;

  // Recompute actual chunk IDs from current positions
  torch::Tensor actual_chunk_ids =
      computeChunkIds(xyz_.index({processed_indices}), chunk_size_);
  torch::Tensor old_chunk_ids = gaussian_chunk_ids_.index({processed_indices});
  torch::Tensor moved_mask = (actual_chunk_ids != old_chunk_ids);

  if (!moved_mask.any().item<bool>()) {
    std::cout << "[Batch Redistribution] No gaussians moved" << std::endl;
    return;
  }

  torch::Tensor moved_local = torch::where(moved_mask)[0];
  torch::Tensor destination_ids = actual_chunk_ids.index({moved_local});
  std::cout << "[Batch Redistribution] " << moved_local.size(0)
            << " gaussians moved to different chunks" << std::endl;

  // Pre-load destination chunks to prevent spillover classification
  torch::Tensor unique_destinations =
      std::get<0>(torch::_unique2(destination_ids));
  if (unique_destinations.size(0) > 0) {
    std::cout << "[Batch Redistribution] Pre-loading "
              << unique_destinations.size(0) << " destination chunks"
              << std::endl;
    loadChunks(unique_destinations);
  }

  // Recompute after loadChunks (eviction may have invalidated indices)
  torch::Tensor updated_mask =
      torch::isin(gaussian_chunk_ids_, processed_chunk_ids);
  torch::Tensor updated_indices = torch::where(updated_mask)[0];

  if (updated_indices.size(0) == 0) {
    std::cout << "[Batch Redistribution] WARNING: All gaussians from processed "
                 "chunks were evicted during loading!" << std::endl;
    return;
  }

  torch::Tensor updated_actual =
      computeChunkIds(xyz_.index({updated_indices}), chunk_size_);
  torch::Tensor updated_old = gaussian_chunk_ids_.index({updated_indices});
  torch::Tensor updated_moved_mask = (updated_actual != updated_old);

  if (!updated_moved_mask.any().item<bool>()) {
    std::cout << "[Batch Redistribution] No gaussians moved after recomputation"
              << std::endl;
    return;
  }

  torch::Tensor final_moved_local = torch::where(updated_moved_mask)[0];
  torch::Tensor final_moved_global = updated_indices.index({final_moved_local});
  torch::Tensor final_destinations = updated_actual.index({final_moved_local});

  std::cout << "[Batch Redistribution] " << final_moved_local.size(0)
            << " gaussians need redistribution" << std::endl;

  gaussian_chunk_ids_.index_put_({final_moved_global}, final_destinations);
}

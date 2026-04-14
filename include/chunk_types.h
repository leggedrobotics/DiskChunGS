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
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <unordered_map>

struct AABB {
  Eigen::Vector3f min;
  Eigen::Vector3f max;

  AABB() : min(Eigen::Vector3f::Zero()), max(Eigen::Vector3f::Zero()) {}
  AABB(const Eigen::Vector3f &min_val, const Eigen::Vector3f &max_val)
      : min(min_val), max(max_val) {}
};

struct ChunkCoord {
  int64_t x, y, z;

  bool operator==(const ChunkCoord &other) const {
    return x == other.x && y == other.y && z == other.z;
  }

  bool operator<(const ChunkCoord &other) const {
    if (x != other.x) return x < other.x;
    if (y != other.y) return y < other.y;
    return z < other.z;
  }
};

// Custom hash function for ChunkCoord
struct ChunkCoordHash {
  std::size_t operator()(const ChunkCoord &coord) const {
    // Simple hash combining function
    std::size_t h1 = std::hash<int>{}(coord.x);
    std::size_t h2 = std::hash<int>{}(coord.y);
    std::size_t h3 = std::hash<int>{}(coord.z);
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};

// Get chunk coordinate from 3D position
inline ChunkCoord getChunkCoord(const Eigen::Vector3f &position,
                                float chunk_size) {
  float half_chunk = chunk_size * 0.5f;
  return ChunkCoord{static_cast<int64_t>(
                        std::floor((position.x() + half_chunk) / chunk_size)),
                    static_cast<int64_t>(
                        std::floor((position.y() + half_chunk) / chunk_size)),
                    static_cast<int64_t>(
                        std::floor((position.z() + half_chunk) / chunk_size))};
}

// Get chunk center
inline Eigen::Vector3f getChunkCenter(const ChunkCoord &coord,
                                      float chunk_size) {
  return Eigen::Vector3f(coord.x * chunk_size, coord.y * chunk_size,
                         coord.z * chunk_size);
}

// Calculate AABB for a chunk
inline AABB getChunkAABB(const ChunkCoord &coord, float chunk_size) {
  float half_chunk = chunk_size * 0.5f;
  Eigen::Vector3f center(coord.x * chunk_size, coord.y * chunk_size,
                         coord.z * chunk_size);
  Eigen::Vector3f min_corner = center - Eigen::Vector3f::Constant(half_chunk);
  Eigen::Vector3f max_corner = center + Eigen::Vector3f::Constant(half_chunk);
  return AABB(min_corner, max_corner);
}

inline AABB getRegionAABB(const ChunkCoord &min_coord,
                          const ChunkCoord &max_coord,
                          float chunk_size) {
  float half_chunk = chunk_size * 0.5f;

  Eigen::Vector3f min_pos(min_coord.x * chunk_size - half_chunk,
                          min_coord.y * chunk_size - half_chunk,
                          min_coord.z * chunk_size - half_chunk);

  Eigen::Vector3f max_pos((max_coord.x + 1) * chunk_size - half_chunk,
                          (max_coord.y + 1) * chunk_size - half_chunk,
                          (max_coord.z + 1) * chunk_size - half_chunk);

  return AABB(min_pos, max_pos);
}

// Encode/decode single chunk coordinates
inline int64_t encodeChunkCoord(const ChunkCoord &coord) {
  const int64_t OFFSET = 1048576;  // 2^20

  int64_t x = coord.x + OFFSET;
  int64_t y = coord.y + OFFSET;
  int64_t z = coord.z + OFFSET;

  // 21 bits per coordinate = 63 total bits
  return x * (1LL << 42) + y * (1LL << 21) + z;
}

inline ChunkCoord decodeChunkCoord(int64_t chunk_id) {
  const int64_t OFFSET = 1048576;        // 2^20
  const int64_t FIELD_SIZE = 1LL << 21;  // 2^21 = 2,097,152

  int64_t z = (chunk_id % FIELD_SIZE) - OFFSET;
  int64_t y = ((chunk_id / FIELD_SIZE) % FIELD_SIZE) - OFFSET;
  int64_t x = (chunk_id / (FIELD_SIZE * FIELD_SIZE)) - OFFSET;

  return ChunkCoord{x, y, z};
}

// Torch tensor versions of chunk utilities

// Encode chunk coordinates tensor to IDs
inline torch::Tensor encodeChunkCoordsTensor(
    const torch::Tensor &chunk_coords) {
  // chunk_coords: [N, 3] with coordinates in range [-1M, +1M]

  // Use 21 bits per coordinate = 2M range = [-1,048,576, +1,048,575] chunks
  // Total: 63 bits used out of 64 available - maximum efficiency
  // Supports ±20,971 km range per axis with 20m chunks
  const int64_t OFFSET = 1048576;  // 2^20

  auto x = chunk_coords.index({torch::indexing::Slice(), 0}) + OFFSET;
  auto y = chunk_coords.index({torch::indexing::Slice(), 1}) + OFFSET;
  auto z = chunk_coords.index({torch::indexing::Slice(), 2}) + OFFSET;

  // Pack: 21 bits each for x, y, z coordinates
  torch::Tensor encoded = x * (1LL << 42) + y * (1LL << 21) + z;

  return encoded;  // Range: 0 to ~9.2 × 10^18
}

// Decode chunk IDs to coordinates tensor
inline torch::Tensor decodeChunkCoordsTensor(const torch::Tensor &encoded_ids) {
  const int64_t OFFSET = 1048576;        // 2^20
  const int64_t FIELD_SIZE = 1LL << 21;  // 2^21 = 2,097,152

  // Extract 21-bit fields using modulo and integer division
  torch::Tensor z = (encoded_ids % FIELD_SIZE) - OFFSET;
  torch::Tensor y = ((encoded_ids / FIELD_SIZE) % FIELD_SIZE) - OFFSET;
  torch::Tensor x = (encoded_ids / (FIELD_SIZE * FIELD_SIZE)) - OFFSET;

  return torch::stack({x, y, z}, /*dim=*/1);
}

// Convert vector of ChunkCoord to tensor
inline torch::Tensor chunkCoordVectorToTensor(
    const std::vector<ChunkCoord> &coords,
    torch::DeviceType device_type = torch::kCUDA) {
  if (coords.empty()) {
    return torch::empty(
        {0, 3},
        torch::TensorOptions().dtype(torch::kInt64).device(device_type));
  }

  // Use from_blob for zero-copy conversion (ChunkCoord is POD with int64_t
  // x,y,z)
  torch::Tensor coord_tensor =
      torch::from_blob(const_cast<ChunkCoord *>(coords.data()),
                       {static_cast<int64_t>(coords.size()), 3},
                       torch::TensorOptions().dtype(torch::kInt64))
          .clone();  // Clone to own the memory

  return coord_tensor.to(device_type);
}

// Compute chunk IDs from 3D positions
inline torch::Tensor computeChunkIds(const torch::Tensor &positions,
                                     float chunk_size) {
  torch::NoGradGuard no_grad;
  float half_chunk = chunk_size * 0.5f;

  torch::Tensor shifted_positions = positions + half_chunk;
  torch::Tensor chunk_coords = torch::floor(shifted_positions / chunk_size);
  chunk_coords = chunk_coords.to(torch::kInt64);

  // Use the SAME encoding as encodeChunkCoordsTensor
  return encodeChunkCoordsTensor(chunk_coords);
}
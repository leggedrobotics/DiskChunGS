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
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

class ProfilingUtils {
 public:
  struct TimingStats {
    double total_time = 0.0;
    size_t call_count = 0;
    double max_time = 0.0;
    double min_time = std::numeric_limits<double>::max();
  };

  static ProfilingUtils& getInstance() {
    static ProfilingUtils instance;
    return instance;
  }

  class Timer {
   public:
    Timer(const std::string& name)
        : name_(name), start_(std::chrono::high_resolution_clock::now()) {}

    void stop() {
      auto end = std::chrono::high_resolution_clock::now();
      double duration =
          std::chrono::duration<double, std::milli>(end - start_).count();
      ProfilingUtils::getInstance().addTiming(name_, duration);
    }

   private:
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  };

  void addTiming(const std::string& name, double duration_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& stats = timings_[name];
    stats.total_time += duration_ms;
    stats.call_count++;
    stats.max_time = std::max(stats.max_time, duration_ms);
    stats.min_time = std::min(stats.min_time, duration_ms);
  }

  void printStats() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::cout << "\n=== Profiling Statistics ===\n";
    std::cout << std::setw(30) << std::left << "Section" << std::setw(15)
              << "Avg (ms)" << std::setw(15) << "Min (ms)" << std::setw(15)
              << "Max (ms)" << std::setw(15) << "Total (ms)" << std::setw(10)
              << "Calls\n";
    std::cout << std::string(100, '-') << "\n";

    for (const auto& pair : timings_) {
      const auto& name = pair.first;
      const auto& stats = pair.second;
      double avg = stats.total_time / stats.call_count;

      std::cout << std::fixed << std::setprecision(2) << std::setw(30)
                << std::left << name << std::setw(15) << avg << std::setw(15)
                << stats.min_time << std::setw(15) << stats.max_time
                << std::setw(15) << stats.total_time << std::setw(10)
                << stats.call_count << "\n";
    }
    std::cout << std::string(100, '-') << "\n";
  }

  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    timings_.clear();
  }

 private:
  ProfilingUtils() = default;
  std::unordered_map<std::string, TimingStats> timings_;
  std::mutex mutex_;
};
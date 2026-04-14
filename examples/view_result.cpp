/**
 * This file is part of Photo-SLAM
 *
 * Copyright (C) 2023-2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2023-2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University
 * of Science and Technology.
 *
 * Photo-SLAM is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * Photo-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * Photo-SLAM. If not, see <http://www.gnu.org/licenses/>.
 */

#include <torch/torch.h>

#include <filesystem>
#include <iostream>
#include <memory>

#include "gaussian_mapper.h"
#include "viewer/imgui_viewer.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << std::endl
              << "Usage: " << argv[0] << " path_to_result_path" /*1*/
              << std::endl;
    return 1;
  }

  // Device
  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }

  // Create GaussianMapper
  std::filesystem::path result_path(argv[1]);
  std::filesystem::path gaussian_cfg_path =
      result_path / "gaussian_mapper_cfg.yaml";
  std::shared_ptr<GaussianMapper> pGausMapper =
      std::make_shared<GaussianMapper>(gaussian_cfg_path, result_path, 0,
                                       device_type);

  // Create Gaussian Viewer
  std::thread viewer_thd;
  std::shared_ptr<ImGuiViewer> pViewer;
  pViewer = std::make_shared<ImGuiViewer>(nullptr, pGausMapper, false);
  pViewer->run();

  return 0;
}

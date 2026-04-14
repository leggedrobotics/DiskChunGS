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

#include "scene/gaussian_keyframe.h"

#include "utils/depth_utils.h"

//==============================================================================
// Static Helper Functions
//==============================================================================

void GaussianKeyframe::transferTensorToDevice(torch::Tensor& tensor,
                                              torch::DeviceType target,
                                              bool restore_grad) {
  if (!tensor.defined()) return;

  bool is_cuda = tensor.device().is_cuda();
  bool want_cuda = (target == torch::kCUDA);

  if (is_cuda != want_cuda) {
    tensor = tensor.to(target);
    if (restore_grad && want_cuda) {
      tensor.requires_grad_(true);
    }
  }
}

void GaussianKeyframe::clearTensor(torch::Tensor& tensor) {
  if (tensor.defined()) {
    tensor.reset();
  }
}

//==============================================================================
// Pose Management
//==============================================================================

void GaussianKeyframe::setPoseImpl(const Eigen::Matrix3d& R,
                                   const Eigen::Vector3d& t) {
  auto tensor_opts = torch::TensorOptions()
                         .dtype(torch::kFloat32)
                         .device(torch::kCUDA)
                         .requires_grad(true);

  rW2C_ = torch::zeros({3, 2}, tensor_opts);
  tW2C_ = torch::zeros({3}, tensor_opts);

  {
    torch::NoGradGuard no_grad;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 2; j++) {
        rW2C_[i][j] = static_cast<float>(R(i, j));
      }
      tW2C_[i] = static_cast<float>(t(i));
    }
  }

  set_pose_ = true;
}

void GaussianKeyframe::setPose(double qw, double qx, double qy, double qz,
                               double tx, double ty, double tz) {
  Eigen::Quaterniond q(qw, qx, qy, qz);
  q.normalize();
  setPoseImpl(q.toRotationMatrix(), Eigen::Vector3d(tx, ty, tz));
}

void GaussianKeyframe::setPose(const Eigen::Quaterniond& q,
                               const Eigen::Vector3d& t) {
  setPoseImpl(q.normalized().toRotationMatrix(), t);
}

Eigen::Matrix3d GaussianKeyframe::tensorToRotationMatrix() const {
  torch::Tensor R_tensor = sixD2RotationMatrix(rW2C_);
  auto R_cpu = R_tensor.detach().cpu();

  Eigen::Matrix3d R_eigen;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      R_eigen(i, j) = R_cpu[i][j].item<float>();
    }
  }
  return R_eigen;
}

Eigen::Vector3d GaussianKeyframe::tensorToTranslation() const {
  auto t_cpu = tW2C_.detach().cpu();

  Eigen::Vector3d t_eigen;
  for (int i = 0; i < 3; i++) {
    t_eigen(i) = t_cpu[i].item<float>();
  }
  return t_eigen;
}

Sophus::SE3d GaussianKeyframe::getPose() {
  Eigen::Matrix3d R = tensorToRotationMatrix();
  Eigen::Vector3d t = tensorToTranslation();
  return Sophus::SE3d(Eigen::Quaterniond(R), t);
}

Sophus::SE3f GaussianKeyframe::getPosef() {
  return getPose().cast<float>();
}

Eigen::Matrix3d GaussianKeyframe::getRotationMatrix() {
  return tensorToRotationMatrix();
}

Eigen::Matrix3f GaussianKeyframe::getRotationMatrixf() {
  return getRotationMatrix().cast<float>();
}

Eigen::Vector3d GaussianKeyframe::getTranslation() {
  return tensorToTranslation();
}

Eigen::Vector3f GaussianKeyframe::getTranslationf() {
  return getTranslation().cast<float>();
}

Eigen::Quaterniond GaussianKeyframe::getQuaternion() {
  return Eigen::Quaterniond(getRotationMatrix());
}

Eigen::Quaternionf GaussianKeyframe::getQuaternionf() {
  return getQuaternion().cast<float>();
}

//==============================================================================
// Tensor Pose Accessors
//==============================================================================

torch::Tensor GaussianKeyframe::sixD2RotationMatrix(
    const torch::Tensor& rW2C) const {
  // 6D representation: first two columns of rotation matrix
  // Recover full rotation via Gram-Schmidt orthogonalization
  auto a1 = rW2C.select(1, 0);
  auto a2 = rW2C.select(1, 1);

  auto b1 = torch::nn::functional::normalize(
      a1, torch::nn::functional::NormalizeFuncOptions().dim(0));

  auto b2 = a2 - torch::sum(b1 * a2) * b1;
  b2 = torch::nn::functional::normalize(
      b2, torch::nn::functional::NormalizeFuncOptions().dim(0));

  auto b3 = torch::cross(b1, b2, 0);

  return torch::stack({b1, b2, b3}, 1);
}

torch::Tensor GaussianKeyframe::getR() {
  return sixD2RotationMatrix(rW2C_);
}

torch::Tensor GaussianKeyframe::getT() {
  return tW2C_;
}

torch::Tensor GaussianKeyframe::getRT() {
  torch::Tensor RT = torch::eye(
      {4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  RT.index_put_({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)},
                getR());
  RT.index_put_({torch::indexing::Slice(0, 3), 3}, getT());
  return RT;
}

torch::Tensor GaussianKeyframe::getCenter() {
  return -getR().transpose(0, 1).mv(getT());
}

//==============================================================================
// Camera Setup
//==============================================================================

void GaussianKeyframe::setCameraParams(const Camera& camera) {
  camera_id_ = camera.camera_id_;
  camera_model_id_ = camera.model_id_;
  image_height_ = camera.height_;
  image_width_ = camera.width_;

  num_gaus_pyramid_sub_levels_ = camera.num_gaus_pyramid_sub_levels_;
  gaus_pyramid_height_ = camera.gaus_pyramid_height_;
  gaus_pyramid_width_ = camera.gaus_pyramid_width_;

  intr_.resize(camera.params_.size());
  for (std::size_t i = 0; i < camera.params_.size(); ++i) {
    intr_[i] = static_cast<float>(camera.params_[i]);
  }

  switch (camera_model_id_) {
    case 1: {  // Pinhole
      float fx = static_cast<float>(camera.params_[0]);
      float fy = static_cast<float>(camera.params_[1]);
      FoVx_ = graphics_utils::focal2fov(fx, camera.width_);
      FoVy_ = graphics_utils::focal2fov(fy, camera.height_);
      set_camera_ = true;
      break;
    }
    default:
      throw std::runtime_error(
          "Colmap camera model not handled: only undistorted datasets "
          "(PINHOLE or SIMPLE_PINHOLE cameras) supported!");
  }
}

void GaussianKeyframe::computeTransformTensors() {
  if (!set_pose_) {
    std::cerr << "Could not compute transform tensors for keyframe " << fid_
              << " because POSE is not set!" << std::endl;
    return;
  }
  if (!set_camera_) {
    std::cerr << "Could not compute transform tensors for keyframe " << fid_
              << " because CAMERA is not set!" << std::endl;
    return;
  }

  world_view_transform_ =
      tensor_utils::EigenMatrix2TorchTensor(getWorld2View2(trans_, scale_),
                                            torch::kCUDA)
          .transpose(0, 1);

  if (!set_projection_matrix_) {
    projection_matrix_ =
        getProjectionMatrix(znear_, zfar_, FoVx_, FoVy_, torch::kCUDA)
            .transpose(0, 1);
    set_projection_matrix_ = true;
  }

  full_proj_transform_ =
      (world_view_transform_.unsqueeze(0).bmm(projection_matrix_.unsqueeze(0)))
          .squeeze(0);

  camera_center_ =
      world_view_transform_.inverse().index({3, torch::indexing::Slice(0, 3)});
}

Eigen::Matrix4f GaussianKeyframe::getWorld2View2(const Eigen::Vector3f& trans,
                                                 float scale) {
  Eigen::Matrix3f R = tensorToRotationMatrix().cast<float>();
  Eigen::Vector3f t = tensorToTranslation().cast<float>();

  Eigen::Matrix4f Rt = Eigen::Matrix4f::Zero();
  Rt.topLeftCorner<3, 3>() = R;
  Rt.topRightCorner<3, 1>() = t;
  Rt(3, 3) = 1.0f;

  Eigen::Matrix4f C2W = Rt.inverse();
  Eigen::Vector3f cam_center = C2W.block<3, 1>(0, 3);
  cam_center = (cam_center + trans) * scale;
  C2W.block<3, 1>(0, 3) = cam_center;

  return C2W.inverse();
}

torch::Tensor GaussianKeyframe::getProjectionMatrix(
    float znear, float zfar, float fovX, float fovY,
    torch::DeviceType device_type) {
  float tanHalfFovY = std::tan(fovY / 2);
  float tanHalfFovX = std::tan(fovX / 2);

  float top = tanHalfFovY * znear;
  float bottom = -top;
  float right = tanHalfFovX * znear;
  float left = -right;

  torch::Tensor P =
      torch::zeros({4, 4}, torch::TensorOptions().device(device_type));

  constexpr float z_sign = 1.0f;

  P.index({0, 0}) = 2.0f * znear / (right - left);
  P.index({1, 1}) = 2.0f * znear / (top - bottom);
  P.index({0, 2}) = (right + left) / (right - left);
  P.index({1, 2}) = (top + bottom) / (top - bottom);
  P.index({3, 2}) = z_sign;
  P.index({2, 2}) = z_sign * zfar / (zfar - znear);
  P.index({2, 3}) = -(zfar * znear) / (zfar - znear);

  return P;
}

//==============================================================================
// 2D/3D Point Correspondences
//==============================================================================

void GaussianKeyframe::setPoints2D(
    const std::vector<Eigen::Vector2d>& points2D) {
  points2D_.clear();
  points2D_.resize(points2D.size());
  for (std::size_t i = 0; i < points2D.size(); ++i) {
    points2D_[i].xy_ = points2D[i];
  }
}

void GaussianKeyframe::setPoint3DIdxForPoint2D(point2D_idx_t point2D_idx,
                                               point3D_id_t point3D_id) {
  points2D_.at(point2D_idx).point3D_id_ = point3D_id;
}

std::tuple<std::vector<float>, std::vector<float>>
GaussianKeyframe::extractValidKeypointsForDepthAlignment() const {
  std::vector<float> valid_pixel_coords;
  std::vector<float> valid_depths;

  assert(kps_pixel_.size() % 2 == 0);
  assert(kps_point_local_.size() % 3 == 0);

  int num_keypoints = kps_pixel_.size() / 2;

  for (int i = 0; i < num_keypoints; i++) {
    float u = kps_pixel_[2 * i];
    float v = kps_pixel_[2 * i + 1];
    float x = kps_point_local_[3 * i];
    float y = kps_point_local_[3 * i + 1];
    float z = kps_point_local_[3 * i + 2];

    bool has_valid_3d = (z > 0.0f) &&
                        (u >= 0 && u < image_width_) &&
                        (v >= 0 && v < image_height_) &&
                        std::isfinite(x) && std::isfinite(y) && std::isfinite(z);

    if (has_valid_3d) {
      valid_pixel_coords.push_back(u);
      valid_pixel_coords.push_back(v);
      valid_depths.push_back(z);
    }
  }

  return {valid_pixel_coords, valid_depths};
}

//==============================================================================
// Depth Estimation Setup
//==============================================================================

void GaussianKeyframe::setupStereoData(
    const cv::Mat& img_undist,
    const cv::Mat& img_auxiliary_undist,
    float baseline,
    torch::DeviceType device_type,
    std::shared_ptr<StereoDepth> depth_estimator,
    float min_depth,
    float max_depth) {
  if (img_auxiliary_undist.empty()) return;

  // Convert to uint8 for stereo matching
  cv::Mat left_img_uint8, right_img_uint8;
  img_undist.convertTo(left_img_uint8, CV_8UC3, 255.0);
  img_auxiliary_undist.convertTo(right_img_uint8, CV_8UC3, 255.0);

  cv::Mat depth = depth_estimator->estimate_metric_depth(
      left_img_uint8, right_img_uint8, intr_[0], baseline);

  // Clamp and invert depth
  constexpr float kMinDepthClamp = 1e-8f;
  cv::max(depth, kMinDepthClamp, depth);

  cv::Mat inverted_depth;
  cv::divide(1.0f, depth, inverted_depth);

  torch::Tensor depth_image =
      tensor_utils::cvMat2TorchTensor_Float32(inverted_depth, torch::kCUDA)
          .unsqueeze(0)
          .unsqueeze(0);

  depth_confidence_ = depth_utils::computeDepthConfidence(depth_image);
  generateInverseDepthPyramid(inverted_depth);
}

void GaussianKeyframe::setupMonoData(
    const cv::Mat& img_undist,
    torch::DeviceType device_type,
    std::shared_ptr<MonoDepth> depth_estimator,
    float min_depth,
    float max_depth) {
  auto [relative_depth, depth_confidence] =
      depth_estimator->estimate_depth(img_undist, intr_[0]);

  depth_confidence_ = depth_confidence;

  auto [valid_pixel_coords, valid_depths] =
      extractValidKeypointsForDepthAlignment();

  if (valid_depths.size() < 5) {
    std::cout << "Not enough valid depths for monocular depth alignment: "
              << valid_depths.size() << std::endl;
    return;
  }

  torch::Tensor aligned_inv_depth = depth_estimator->align_depth(
      relative_depth, valid_pixel_coords, valid_depths,
      image_width_, image_height_);

  torch::Tensor inv_depth =
      torch::nn::functional::interpolate(
          aligned_inv_depth,
          torch::nn::functional::InterpolateFuncOptions()
              .size(std::vector<int64_t>{image_height_, image_width_})
              .mode(torch::kBilinear)
              .align_corners(true))
          .squeeze(0)
          .squeeze(0);

  cv::Mat inverted_depth_mat =
      tensor_utils::torchTensor2CvMat_Float32(inv_depth);
  generateInverseDepthPyramid(inverted_depth_mat);
}

void GaussianKeyframe::setupRGBDData(const cv::Mat& img_auxiliary_undist) {
  cv::Mat clamped_depth;
  cv::max(img_auxiliary_undist, 1e-8, clamped_depth);

  cv::Mat inverse_depth;
  cv::divide(1.0, clamped_depth, inverse_depth);

  torch::Tensor depth_image =
      tensor_utils::cvMat2TorchTensor_Float32(inverse_depth, torch::kCUDA)
          .unsqueeze(0)
          .unsqueeze(0);

  depth_confidence_ = depth_utils::computeDepthConfidence(depth_image);
  generateInverseDepthPyramid(inverse_depth);
}

//==============================================================================
// Image Pyramid Generation
//==============================================================================

void GaussianKeyframe::generateImagePyramid(const cv::Mat& img_undist) {
  assert(!img_undist.empty());

  cv::cuda::GpuMat img_gpu;
  img_gpu.upload(img_undist);
  gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);

  for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
    cv::cuda::GpuMat img_resized;
    cv::cuda::resize(img_gpu, img_resized,
                     cv::Size(gaus_pyramid_width_[l], gaus_pyramid_height_[l]));
    gaus_pyramid_original_image_[l] =
        tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
  }
}

void GaussianKeyframe::generateInverseDepthPyramid(const cv::Mat& depth_mat) {
  if (depth_mat.empty()) return;

  gaus_pyramid_inv_depth_image_.resize(num_gaus_pyramid_sub_levels_);

  cv::cuda::GpuMat depth_gpu;
  depth_gpu.upload(depth_mat);

  for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
    cv::cuda::GpuMat depth_resized;
    cv::cuda::resize(
        depth_gpu, depth_resized,
        cv::Size(gaus_pyramid_width_[l], gaus_pyramid_height_[l]),
        0, 0, cv::INTER_NEAREST);
    gaus_pyramid_inv_depth_image_[l] =
        tensor_utils::cvGpuMat2TorchTensor_Float32(depth_resized);
  }
}

int GaussianKeyframe::getCurrentGausPyramidLevel() {
  // Start from highest level (smallest image) and work down
  for (int i = num_gaus_pyramid_sub_levels_ - 1; i >= 0; --i) {
    if (gaus_pyramid_times_of_use_[i]) {
      --gaus_pyramid_times_of_use_[i];
      return i;
    }
  }
  return 0;  // Default to full resolution
}

//==============================================================================
// Training Data Access
//==============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int, int>
GaussianKeyframe::getTrainingData(
    const torch::Tensor& undistort_mask,
    const std::vector<torch::Tensor>& pyramid_masks) {
  int level = getCurrentGausPyramidLevel();

  int height = gaus_pyramid_height_[level];
  int width = gaus_pyramid_width_[level];
  torch::Tensor gt_image = gaus_pyramid_original_image_[level].cuda();
  torch::Tensor mask = pyramid_masks[level];

  torch::Tensor gt_inv_depth;
  if (!gaus_pyramid_inv_depth_image_.empty() &&
      level < static_cast<int>(gaus_pyramid_inv_depth_image_.size())) {
    gt_inv_depth = gaus_pyramid_inv_depth_image_[level].cuda();
    gt_inv_depth = gt_inv_depth * depth_scale_ + depth_bias_;
  }

  return {gt_image, gt_inv_depth, mask, height, width};
}

//==============================================================================
// Optimization
//==============================================================================

void GaussianKeyframe::initOptimizer(torch::DeviceType device_type,
                                     float pose_lr,
                                     float exposure_lr,
                                     float depth_scale_bias_lr) {
  pose_lr_ = pose_lr;

  auto tensor_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(device_type);

  // Initialize exposure as identity transform [I|0]
  exposure_transform_ = torch::zeros({3, 4}, tensor_opts);
  exposure_transform_.slice(1, 0, 3) = torch::eye(3, tensor_opts);
  exposure_transform_.requires_grad_(true);

  depth_scale_ = torch::ones({1}, tensor_opts);
  depth_scale_.requires_grad_(true);

  depth_bias_ = torch::zeros({1}, tensor_opts);
  depth_bias_.requires_grad_(true);

  Tensor_vec_rW2C_ = {rW2C_};
  Tensor_vec_tW2C_ = {tW2C_};
  Tensor_vec_exposure_ = {exposure_transform_};
  Tensor_vec_depth_scale_ = {depth_scale_};
  Tensor_vec_depth_bias_ = {depth_bias_};

  torch::optim::AdamOptions adam_options(pose_lr);
  optimizer_ = std::make_shared<torch::optim::Adam>(Tensor_vec_rW2C_,
                                                    adam_options);

  optimizer_->add_param_group(Tensor_vec_tW2C_);
  optimizer_->param_groups()[1].options().set_lr(pose_lr);

  optimizer_->add_param_group(Tensor_vec_exposure_);
  optimizer_->param_groups()[2].options().set_lr(exposure_lr);

  optimizer_->add_param_group(Tensor_vec_depth_scale_);
  optimizer_->param_groups()[3].options().set_lr(depth_scale_bias_lr);

  optimizer_->add_param_group(Tensor_vec_depth_bias_);
  optimizer_->param_groups()[4].options().set_lr(depth_scale_bias_lr);
}

void GaussianKeyframe::step() {
  if (!optimizer_) return;
  optimizer_->step();
  optimizer_->zero_grad();
}

torch::Tensor GaussianKeyframe::applyExposureTransform(torch::Tensor& colors) {
  if (!exposure_transform_.defined()) return colors;

  // [C, H, W] -> [H, W, C]
  auto colors_hwc = colors.permute({1, 2, 0});
  auto original_shape = colors_hwc.sizes();

  // Flatten to [H*W, 3] for matrix multiplication
  auto colors_flat = colors_hwc.view({-1, 3});

  // Extract 3x3 transform and bias from 3x4 matrix
  auto transform_3x3 = exposure_transform_.slice(1, 0, 3);
  auto bias = exposure_transform_.slice(1, 3, 4).squeeze(1);

  // Apply affine transform
  auto transformed = torch::mm(colors_flat, transform_3x3.t()) + bias.unsqueeze(0);

  // Reshape back to [C, H, W]
  return transformed.view(original_shape).permute({2, 0, 1}).clamp(0.0f, 1.0f);
}

//==============================================================================
// Memory Management (Disk Serialization)
//==============================================================================

void GaussianKeyframe::saveDataToDisk() {
  if (!loaded_) {
    throw std::runtime_error("Can't save keyframe to disk that isn't loaded");
  }

  if (!on_disk_) {
    std::filesystem::create_directories(keyframe_save_dir_);

    torch::serialize::OutputArchive archive;

    if (depth_confidence_.defined()) {
      archive.write("depth_confidence_", depth_confidence_);
    }

    if (feature_map_.defined()) {
      archive.write("feature_map_", feature_map_);
    }

    if (!gaus_pyramid_original_image_.empty()) {
      archive.write("pyramid_size",
                    torch::tensor(static_cast<int64_t>(
                        gaus_pyramid_original_image_.size())));
      for (size_t i = 0; i < gaus_pyramid_original_image_.size(); ++i) {
        if (gaus_pyramid_original_image_[i].defined()) {
          archive.write("pyramid_image_" + std::to_string(i),
                        gaus_pyramid_original_image_[i]);
        }
      }
    }

    if (!gaus_pyramid_inv_depth_image_.empty()) {
      archive.write("pyramid_depth_size",
                    torch::tensor(static_cast<int64_t>(
                        gaus_pyramid_inv_depth_image_.size())));
      for (size_t i = 0; i < gaus_pyramid_inv_depth_image_.size(); ++i) {
        if (gaus_pyramid_inv_depth_image_[i].defined()) {
          archive.write("pyramid_depth_" + std::to_string(i),
                        gaus_pyramid_inv_depth_image_[i]);
        }
      }
    }

    std::filesystem::path data_path =
        keyframe_save_dir_ / ("keyframe_data_" + std::to_string(fid_) + ".pt");
    archive.save_to(data_path.string());

    on_disk_ = true;
  }

  // Clear tensors from memory
  clearTensor(depth_confidence_);
  clearTensor(feature_map_);

  for (auto& img : gaus_pyramid_original_image_) {
    clearTensor(img);
  }
  gaus_pyramid_original_image_.clear();

  for (auto& depth : gaus_pyramid_inv_depth_image_) {
    clearTensor(depth);
  }
  gaus_pyramid_inv_depth_image_.clear();

  loaded_ = false;
}

void GaussianKeyframe::loadDataFromDisk() {
  if (loaded_) {
    std::cout << "WARN: Loading keyframe that is already marked as loaded!"
              << std::endl;
  }

  std::filesystem::path data_path =
      keyframe_save_dir_ / ("keyframe_data_" + std::to_string(fid_) + ".pt");

  if (!std::filesystem::exists(data_path)) {
    std::cout << "No data file found for keyframe " << fid_ << " at "
              << data_path << std::endl;
    return;
  }

  torch::serialize::InputArchive archive;

  try {
    archive.load_from(data_path.string());

    archive.read("depth_confidence_", depth_confidence_);
    depth_confidence_ = depth_confidence_.to(torch::kCUDA);

    archive.read("feature_map_", feature_map_);
    feature_map_ = feature_map_.to(torch::kCUDA);

    torch::Tensor pyramid_size_tensor;
    archive.read("pyramid_size", pyramid_size_tensor);
    int pyramid_size = pyramid_size_tensor.item<int64_t>();

    gaus_pyramid_original_image_.resize(pyramid_size);
    for (int i = 0; i < pyramid_size; ++i) {
      archive.read("pyramid_image_" + std::to_string(i),
                   gaus_pyramid_original_image_[i]);
      gaus_pyramid_original_image_[i] =
          gaus_pyramid_original_image_[i].to(torch::kCUDA);
    }

    torch::Tensor pyramid_depth_size_tensor;
    archive.read("pyramid_depth_size", pyramid_depth_size_tensor);
    int pyramid_depth_size = pyramid_depth_size_tensor.item<int64_t>();

    gaus_pyramid_inv_depth_image_.resize(pyramid_depth_size);
    for (int i = 0; i < pyramid_depth_size; ++i) {
      archive.read("pyramid_depth_" + std::to_string(i),
                   gaus_pyramid_inv_depth_image_[i]);
      gaus_pyramid_inv_depth_image_[i] =
          gaus_pyramid_inv_depth_image_[i].to(torch::kCUDA);
    }

    loaded_ = true;
  } catch (const std::exception& e) {
    std::cerr << "Error loading data for keyframe " << fid_ << ": " << e.what()
              << std::endl;
    loaded_ = false;
    throw;
  }
}

void GaussianKeyframe::transferToCPU() {
  if (!loaded_) {
    std::cout << "WARN: Tried to transfer keyframe to CPU that isn't loaded!"
              << std::endl;
    return;
  }

  transferTensorToDevice(depth_confidence_, torch::kCPU);
  transferTensorToDevice(feature_map_, torch::kCPU);

  for (auto& img : gaus_pyramid_original_image_) {
    transferTensorToDevice(img, torch::kCPU);
  }

  for (auto& depth : gaus_pyramid_inv_depth_image_) {
    transferTensorToDevice(depth, torch::kCPU);
  }

  transferTensorToDevice(rW2C_, torch::kCPU);
  transferTensorToDevice(tW2C_, torch::kCPU);
  transferTensorToDevice(exposure_transform_, torch::kCPU);
  transferTensorToDevice(depth_scale_, torch::kCPU);
  transferTensorToDevice(depth_bias_, torch::kCPU);

  loaded_ = false;
}

void GaussianKeyframe::transferToGPU() {
  if (loaded_) {
    std::cout << "WARN: Tried to transfer keyframe to GPU that's already loaded!"
              << std::endl;
    return;
  }

  transferTensorToDevice(depth_confidence_, torch::kCUDA);
  transferTensorToDevice(feature_map_, torch::kCUDA);

  for (auto& img : gaus_pyramid_original_image_) {
    transferTensorToDevice(img, torch::kCUDA);
  }

  for (auto& depth : gaus_pyramid_inv_depth_image_) {
    transferTensorToDevice(depth, torch::kCUDA);
  }

  // Pose and optimization params need gradients restored
  transferTensorToDevice(rW2C_, torch::kCUDA, /*restore_grad=*/true);
  transferTensorToDevice(tW2C_, torch::kCUDA, /*restore_grad=*/true);
  transferTensorToDevice(exposure_transform_, torch::kCUDA, /*restore_grad=*/true);
  transferTensorToDevice(depth_scale_, torch::kCUDA, /*restore_grad=*/true);
  transferTensorToDevice(depth_bias_, torch::kCUDA, /*restore_grad=*/true);

  // Update optimizer param groups
  if (optimizer_) {
    Tensor_vec_rW2C_ = {rW2C_};
    Tensor_vec_tW2C_ = {tW2C_};
    Tensor_vec_exposure_ = {exposure_transform_};
    Tensor_vec_depth_scale_ = {depth_scale_};
    Tensor_vec_depth_bias_ = {depth_bias_};
  }

  loaded_ = true;
}
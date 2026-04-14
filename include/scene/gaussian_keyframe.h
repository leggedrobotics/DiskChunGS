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

#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/torch.h>

#include <Eigen/Geometry>
#include <memory>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"
#include "camera.h"
#include "depth/mono_depth.h"
#include "depth/stereo_depth.h"
#include "geometry/point2d.h"
#include "types.h"
#include "utils/debugging_utils.h"
#include "utils/general_utils.h"
#include "utils/graphics_utils.h"
#include "utils/tensor_utils.h"

/**
 * @brief Represents a keyframe in the Gaussian Splatting SLAM system.
 *
 * A GaussianKeyframe stores camera pose, intrinsics, image pyramids, and depth
 * data needed for training and rendering Gaussian splats. It supports
 * optimizable pose and exposure parameters, and can be serialized to/from disk
 * for memory management.
 */
class GaussianKeyframe {
 public:
  //============================================================================
  // Construction
  //============================================================================

  GaussianKeyframe() = default;

  /**
   * @brief Constructs a keyframe with the given frame ID.
   * @param fid Frame identifier
   * @param creation_iter Iteration at which the keyframe was created
   * @param keyframe_save_dir Directory for saving keyframe data to disk
   */
  explicit GaussianKeyframe(std::size_t fid,
                            int creation_iter = 0,
                            std::filesystem::path keyframe_save_dir = "")
      : fid_(fid),
        creation_iter_(creation_iter),
        keyframe_save_dir_(keyframe_save_dir) {}

  //============================================================================
  // Pose Management
  //============================================================================

  /**
   * @brief Sets the camera pose from quaternion and translation components.
   * @param qw, qx, qy, qz Quaternion components (will be normalized)
   * @param tx, ty, tz Translation components
   */
  void setPose(double qw, double qx, double qy, double qz,
               double tx, double ty, double tz);

  /**
   * @brief Sets the camera pose from Eigen quaternion and translation.
   * @param q Rotation quaternion (will be normalized)
   * @param t Translation vector
   */
  void setPose(const Eigen::Quaterniond& q, const Eigen::Vector3d& t);

  /** @brief Gets the camera pose as an SE3 transformation (double precision). */
  Sophus::SE3d getPose();

  /** @brief Gets the camera pose as an SE3 transformation (single precision). */
  Sophus::SE3f getPosef();

  /** @brief Gets the rotation matrix (double precision). */
  Eigen::Matrix3d getRotationMatrix();

  /** @brief Gets the rotation matrix (single precision). */
  Eigen::Matrix3f getRotationMatrixf();

  /** @brief Gets the translation vector (double precision). */
  Eigen::Vector3d getTranslation();

  /** @brief Gets the translation vector (single precision). */
  Eigen::Vector3f getTranslationf();

  /** @brief Gets the rotation as a quaternion (double precision). */
  Eigen::Quaterniond getQuaternion();

  /** @brief Gets the rotation as a quaternion (single precision). */
  Eigen::Quaternionf getQuaternionf();

  //============================================================================
  // Tensor Pose Accessors (for optimization/rendering)
  //============================================================================

  /** @brief Gets the rotation matrix as a torch tensor. */
  torch::Tensor getR();

  /** @brief Gets the translation as a torch tensor. */
  torch::Tensor getT();

  /** @brief Gets the 4x4 rigid transformation matrix [R|t] as a tensor. */
  torch::Tensor getRT();

  /** @brief Gets the camera center in world coordinates. */
  torch::Tensor getCenter();

  //============================================================================
  // Camera Setup
  //============================================================================

  /**
   * @brief Sets camera parameters from a Camera object.
   * @param camera Camera containing intrinsics and image dimensions
   */
  void setCameraParams(const Camera& camera);

  /** @brief Computes world-view, projection, and full projection transforms. */
  void computeTransformTensors();

  /**
   * @brief Computes the world-to-view matrix with optional translation/scale.
   * @param trans Additional translation to apply
   * @param scale Scale factor
   * @return 4x4 world-to-view transformation matrix
   */
  Eigen::Matrix4f getWorld2View2(const Eigen::Vector3f& trans = {0.0f, 0.0f,
                                                                 0.0f},
                                 float scale = 1.0f);

  /**
   * @brief Computes an OpenGL-style projection matrix.
   * @param znear Near clipping plane
   * @param zfar Far clipping plane
   * @param fovX Horizontal field of view (radians)
   * @param fovY Vertical field of view (radians)
   * @param device_type Device to create the tensor on
   * @return 4x4 projection matrix
   */
  torch::Tensor getProjectionMatrix(float znear, float zfar,
                                    float fovX, float fovY,
                                    torch::DeviceType device_type = torch::kCUDA);

  //============================================================================
  // 2D/3D Point Correspondences
  //============================================================================

  /**
   * @brief Sets the 2D keypoints observed in this frame.
   * @param points2D Vector of 2D point coordinates
   */
  void setPoints2D(const std::vector<Eigen::Vector2d>& points2D);

  /**
   * @brief Associates a 2D point with a 3D point ID.
   * @param point2D_idx Index of the 2D point
   * @param point3D_id ID of the corresponding 3D point
   */
  void setPoint3DIdxForPoint2D(point2D_idx_t point2D_idx,
                               point3D_id_t point3D_id);

  /**
   * @brief Extracts keypoints with valid 3D correspondences for depth alignment.
   * @return Tuple of (pixel_coordinates, depths) for valid keypoints
   */
  std::tuple<std::vector<float>, std::vector<float>>
  extractValidKeypointsForDepthAlignment() const;

  //============================================================================
  // Depth Estimation Setup
  //============================================================================

  /**
   * @brief Initializes depth data from stereo images.
   * @param img_undist Undistorted left/main image
   * @param img_auxiliary_undist Undistorted right/auxiliary image
   * @param baseline Stereo baseline in meters
   * @param device_type Device for tensor operations
   * @param depth_estimator Stereo depth estimation module
   * @param min_depth Minimum valid depth
   * @param max_depth Maximum valid depth
   */
  void setupStereoData(const cv::Mat& img_undist,
                       const cv::Mat& img_auxiliary_undist,
                       float baseline,
                       torch::DeviceType device_type,
                       std::shared_ptr<StereoDepth> depth_estimator,
                       float min_depth,
                       float max_depth);

  /**
   * @brief Initializes depth data from monocular depth estimation.
   * @param img_undist Undistorted image
   * @param device_type Device for tensor operations
   * @param depth_estimator Monocular depth estimation module
   * @param min_depth Minimum valid depth
   * @param max_depth Maximum valid depth
   */
  void setupMonoData(const cv::Mat& img_undist,
                     torch::DeviceType device_type,
                     std::shared_ptr<MonoDepth> depth_estimator,
                     float min_depth,
                     float max_depth);

  /**
   * @brief Initializes depth data from RGB-D input.
   * @param img_auxiliary_undist Depth image (float, meters)
   */
  void setupRGBDData(const cv::Mat& img_auxiliary_undist);

  //============================================================================
  // Image Pyramid Generation
  //============================================================================

  /**
   * @brief Generates a multi-resolution image pyramid for coarse-to-fine training.
   * @param img_undist Undistorted input image
   */
  void generateImagePyramid(const cv::Mat& img_undist);

  /**
   * @brief Generates a multi-resolution inverse depth pyramid.
   * @param depth_mat Inverse depth image (float)
   */
  void generateInverseDepthPyramid(const cv::Mat& depth_mat);

  /**
   * @brief Gets the current pyramid level for training.
   * @return Pyramid level index (0 = full resolution)
   */
  int getCurrentGausPyramidLevel();

  //============================================================================
  // Training Data Access
  //============================================================================

  /**
   * @brief Retrieves training data at the current pyramid level.
   * @param undistort_mask Mask for valid pixels
   * @param pyramid_masks Masks for each pyramid level
   * @return Tuple of (gt_image, gt_inv_depth, mask, height, width)
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int, int>
  getTrainingData(const torch::Tensor& undistort_mask,
                  const std::vector<torch::Tensor>& pyramid_masks);

  //============================================================================
  // Optimization
  //============================================================================

  /**
   * @brief Initializes the Adam optimizer for pose and exposure parameters.
   * @param device_type Device for tensor operations
   * @param pose_lr Learning rate for pose parameters
   * @param exposure_lr Learning rate for exposure transform
   * @param depth_scale_bias_lr Learning rate for depth scale/bias
   */
  void initOptimizer(torch::DeviceType device_type,
                     float pose_lr,
                     float exposure_lr,
                     float depth_scale_bias_lr);

  /** @brief Performs one optimization step and zeros gradients. */
  void step();

  /**
   * @brief Applies learned exposure/color correction to rendered colors.
   * @param colors Input color tensor [C, H, W]
   * @return Corrected color tensor, clamped to [0, 1]
   */
  torch::Tensor applyExposureTransform(torch::Tensor& colors);

  //============================================================================
  // Memory Management (Disk Serialization)
  //============================================================================

  /** @brief Saves memory-intensive data to disk and clears from GPU memory. */
  void saveDataToDisk();

  /** @brief Loads previously saved data from disk back into GPU memory. */
  void loadDataFromDisk();

  /** @brief Transfers tensor data from GPU to CPU memory. */
  void transferToCPU();

  /** @brief Transfers tensor data from CPU to GPU memory. */
  void transferToGPU();

 private:
  //============================================================================
  // Private Helper Methods
  //============================================================================

  void setPoseImpl(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);
  Eigen::Matrix3d tensorToRotationMatrix() const;
  Eigen::Vector3d tensorToTranslation() const;

  /**
   * @brief Converts 6D rotation representation to a 3x3 rotation matrix.
   *
   * Uses Gram-Schmidt orthogonalization to recover a valid rotation matrix
   * from the first two columns stored in rW2C_.
   *
   * @param rW2C 3x2 tensor containing first two columns of rotation
   * @return 3x3 rotation matrix tensor
   */
  torch::Tensor sixD2RotationMatrix(const torch::Tensor& rW2C) const;

  /** @brief Transfers a tensor to the target device, optionally restoring gradients. */
  static void transferTensorToDevice(torch::Tensor& tensor,
                                     torch::DeviceType target,
                                     bool restore_grad = false);

  /** @brief Clears a tensor and releases its memory. */
  static void clearTensor(torch::Tensor& tensor);

 public:
  //============================================================================
  // Member Variables - Identification
  //============================================================================

  std::size_t fid_ = 0;             ///< Frame identifier
  int creation_iter_ = 0;           ///< Training iteration when keyframe was created
  int remaining_times_of_use_ = 0;  ///< Counter for training usage

  //============================================================================
  // Member Variables - Camera Parameters
  //============================================================================

  bool set_camera_ = false;   ///< Whether camera parameters have been set
  camera_id_t camera_id_;     ///< Camera identifier
  int camera_model_id_ = 0;   ///< Camera model type (1 = Pinhole)

  std::string img_filename_;  ///< Source image filename
  int image_width_ = 0;       ///< Image width in pixels
  int image_height_ = 0;      ///< Image height in pixels

  std::vector<float> intr_;   ///< Camera intrinsic parameters [fx, fy, cx, cy]
  float FoVx_ = 0.0f;         ///< Horizontal field of view (radians)
  float FoVy_ = 0.0f;         ///< Vertical field of view (radians)

  float znear_ = 0.01f;       ///< Near clipping plane
  float zfar_ = 100.0f;       ///< Far clipping plane

  //============================================================================
  // Member Variables - Pose (Optimizable)
  //============================================================================

  bool set_pose_ = false;     ///< Whether pose has been set
  torch::Tensor rW2C_;        ///< 3x2 rotation in 6D representation
  torch::Tensor tW2C_;        ///< 3x1 translation vector

  Eigen::Vector3f trans_ = {0.0f, 0.0f, 0.0f};  ///< Additional translation offset
  float scale_ = 1.0f;                          ///< Scale factor for world-to-view

  //============================================================================
  // Member Variables - Transform Tensors (Computed)
  //============================================================================

  bool set_projection_matrix_ = false;  ///< Whether projection matrix is cached
  torch::Tensor world_view_transform_;  ///< 4x4 world-to-view matrix
  torch::Tensor projection_matrix_;     ///< 4x4 projection matrix
  torch::Tensor full_proj_transform_;   ///< Combined world-view-projection
  torch::Tensor camera_center_;         ///< Camera center in world coordinates

  //============================================================================
  // Member Variables - 2D/3D Correspondences
  //============================================================================

  std::vector<Point2D> points2D_;       ///< 2D keypoints in this frame
  std::vector<float> kps_pixel_;        ///< Flattened keypoint pixel coords [u0,v0,u1,v1,...]
  std::vector<float> kps_point_local_;  ///< Flattened 3D points in camera frame [x0,y0,z0,...]

  //============================================================================
  // Member Variables - Image Pyramid
  //============================================================================

  int num_gaus_pyramid_sub_levels_ = 0;             ///< Number of pyramid levels
  std::vector<int> gaus_pyramid_times_of_use_;      ///< Usage count per level
  std::vector<std::size_t> gaus_pyramid_width_;     ///< Width at each level
  std::vector<std::size_t> gaus_pyramid_height_;    ///< Height at each level
  std::vector<torch::Tensor> gaus_pyramid_original_image_;   ///< RGB at each level
  std::vector<torch::Tensor> gaus_pyramid_inv_depth_image_;  ///< Inverse depth at each level

  //============================================================================
  // Member Variables - Optimization Parameters
  //============================================================================

  bool pose_optimization_enabled_ = false;      ///< Whether pose is being optimized
  bool exposure_optimization_enabled_ = false;  ///< Whether exposure is being optimized
  float pose_lr_ = 1e-4f;                       ///< Pose learning rate

  torch::Tensor exposure_transform_;  ///< 3x4 affine color correction matrix
  torch::Tensor depth_scale_;         ///< Learnable depth scale factor
  torch::Tensor depth_bias_;          ///< Learnable depth bias offset

  // Tensor vectors for Adam optimizer param groups
  std::vector<torch::Tensor> Tensor_vec_rW2C_;
  std::vector<torch::Tensor> Tensor_vec_tW2C_;
  std::vector<torch::Tensor> Tensor_vec_exposure_;
  std::vector<torch::Tensor> Tensor_vec_depth_scale_;
  std::vector<torch::Tensor> Tensor_vec_depth_bias_;

  std::shared_ptr<torch::optim::Adam> optimizer_;  ///< Adam optimizer instance

  //============================================================================
  // Member Variables - Feature/Depth Data
  //============================================================================

  torch::Tensor feature_map_;       ///< Encoder feature map for this keyframe
  torch::Tensor depth_confidence_;  ///< Per-pixel depth confidence weights

  //============================================================================
  // Member Variables - Disk Persistence
  //============================================================================

  std::filesystem::path keyframe_save_dir_;  ///< Directory for disk storage
  bool loaded_ = false;       ///< Whether data is currently in GPU memory
  bool on_disk_ = false;      ///< Whether data has been saved to disk
  bool allow_eviction_ = false;  ///< Whether this keyframe can be evicted to disk
};
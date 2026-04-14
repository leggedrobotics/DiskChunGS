/**
 * This file is part of DiskChunGS.
 *
 * Copyright (C) 2023-2024 Longwei Li, Hui Cheng (Photo-SLAM)
 * Copyright (C) 2024 Dapeng Feng (CaRtGS)
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

#include <jsoncpp/json/json.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"
#include "ORB-SLAM3/include/MapDrawer.h"
#include "ORB-SLAM3/include/System.h"
#include "chunk_types.h"
#include "depth/guided_mvs.h"
#include "depth/mono_depth.h"
#include "depth/stereo_depth.h"
#include "depth/stereo_vision.h"
#include "gaussian_mapper_external.h"
#include "geometry/operate_points.h"
#include "scene/gaussian_keyframe.h"
#include "scene/gaussian_scene.h"
#include "scene/keyframe_selection.h"
#include "third_party/xfeat_cpp/include/XFeat.h"
#include "utils/tensor_utils.h"

class KeyframeSelector;  // Forward declaration
class TrajectoryViewer;  // Forward declaration

/**
 * @brief Type alias for keyframe data tuple from ORB-SLAM
 *
 * Contains: (keyframe_id, camera_id, pose, rgb_image, is_loop_closure,
 *            auxiliary_image, keypoint_pixels, keypoint_3d_points, filename)
 */
using KeyframeTuple = std::tuple<unsigned long,
                                 unsigned long,
                                 Sophus::SE3f,
                                 cv::Mat,
                                 bool,
                                 cv::Mat,
                                 std::vector<float>,
                                 std::vector<float>,
                                 std::string>;

#define CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(dir)                 \
  if (!dir.empty() && !std::filesystem::exists(dir))                  \
    if (!std::filesystem::create_directories(dir))                    \
      throw std::runtime_error("Cannot create result directory at " + \
                               dir.string());

/**
 * @brief Parameters for image undistortion and camera calibration
 *
 * Stores the original image size and distortion coefficients needed
 * to perform image undistortion for camera calibration.
 */
struct UndistortParams {
  UndistortParams() : old_size_(0, 0) {
    dist_coeff_ = (cv::Mat_<float>(1, 4) << 0.0f, 0.0f, 0.0f, 0.0f);
  }

  UndistortParams(
      const cv::Size &old_size,
      cv::Mat dist_coeff = (cv::Mat_<float>(1, 4) << 0.0f, 0.0f, 0.0f, 0.0f))
      : old_size_(old_size) {
    dist_coeff.copyTo(dist_coeff_);
  }

  cv::Size old_size_;
  cv::Mat dist_coeff_;  ///< Distortion coefficients [k1, k2, p1, p2]
};

/**
 * @brief Sensor type enumeration for the mapping system
 *
 * Defines supported sensor configurations for visual mapping.
 */
enum SystemSensorType { INVALID = 0, MONOCULAR = 1, STEREO = 2, RGBD = 3 };

/**
 * @brief Recursively copy a directory and all its contents
 *
 * @param source Source directory path to copy from
 * @param destination Destination directory path to copy to
 */
void copyFolder(const std::filesystem::path &source,
                const std::filesystem::path &destination);

/**
 * @brief Main Gaussian Splatting mapper class
 *
 * Supports monocular, stereo, and RGB-D sensors.
 * Manages incremental mapping, loop closure handling, and chunk-based memory.
 */
class GaussianMapper {
 public:
  // ========== Constructors ==========

  /**
   * @brief Construct mapper with ORB-SLAM integration
   *
   * @param pSLAM Shared pointer to ORB-SLAM3 system
   * @param gaussian_config_file_path Path to configuration YAML file
   * @param result_dir Output directory for results and visualizations
   * @param seed Random seed for reproducibility
   * @param device_type Torch device type (CUDA or CPU)
   * @param sensor_type Sensor configuration (MONOCULAR, STEREO, RGBD)
   * @param orb_settings_path Path to ORB-SLAM settings file
   */
  GaussianMapper(
      std::shared_ptr<ORB_SLAM3::System> pSLAM,
      std::filesystem::path gaussian_config_file_path,
      std::filesystem::path result_dir,
      int seed = 0,
      torch::DeviceType device_type = torch::kCUDA,
      ORB_SLAM3::System::eSensor sensor_type = ORB_SLAM3::System::MONOCULAR,
      const string &orb_settings_path = "");

  /**
   * @brief Construct mapper for external pose mode (without ORB-SLAM)
   *
   * @param gaussian_config_file_path Path to configuration YAML file
   * @param result_dir Output directory for results and visualizations
   * @param seed Random seed for reproducibility
   * @param device_type Torch device type (CUDA or CPU)
   */
  GaussianMapper(std::filesystem::path gaussian_config_file_path,
                 std::filesystem::path result_dir,
                 int seed = 0,
                 torch::DeviceType device_type = torch::kCUDA);

  // ========== Core Methods ==========

  /**
   * @brief Read configuration parameters from YAML file
   *
   * Loads all mapper, model, optimization, and pipeline parameters from
   * a configuration file using OpenCV FileStorage. Initializes model
   * parameters, optimization settings, recording options, and viewer settings.
   *
   * @param cfg_path Path to the configuration YAML file
   */
  void readConfigFromFile(std::filesystem::path cfg_path);

  /**
   * @brief Main mapping loop - runs initial and incremental mapping phases
   *
   * Coordinates with ORB-SLAM to process keyframes, perform Gaussian
   * optimization, and handle loop closures. Runs until SLAM shutdown or
   * external stop signal.
   */
  void run();

  /**
   * @brief Execute one iteration of Gaussian optimization
   *
   * Selects a keyframe, renders Gaussians, computes losses (L1, SSIM, depth),
   * performs backpropagation, and updates Gaussian parameters.
   */
  void trainForOneIteration();

  /**
   * @brief Check if mapper has been signaled to stop
   * @return True if stopped, false otherwise
   */
  bool isStopped();

  /**
   * @brief Signal the mapper to stop execution
   * @param going_to_stop Stop flag (default: true)
   */
  void signalStop(const bool going_to_stop = true);

  // ========== Rendering ==========

  /**
   * @brief Render RGB and depth images from a given camera pose
   *
   * @param Tcw Camera-to-world transformation
   * @param width Output image width
   * @param height Output image height
   * @param main_vision Use main vision camera parameters if true
   * @return Tuple of (RGB image, depth image)
   */
  std::tuple<cv::Mat, cv::Mat> renderFromPose(const Sophus::SE3f &Tcw,
                                              const int width,
                                              const int height,
                                              const bool main_vision = false);

  // ========== Training Progress ==========

  /**
   * @brief Get current training iteration number
   * @return Current iteration count
   */
  int getIteration();

  /**
   * @brief Increment iteration counter
   * @param inc Amount to increment (can be negative)
   */
  void increaseIteration(const int inc = 1);

  // ========== Learning Rate Getters ==========

  /** @brief Get initial learning rate for Gaussian positions */
  float positionLearningRateInit();

  /** @brief Get learning rate for spherical harmonic features */
  float featureLearningRate();

  /** @brief Get learning rate for Gaussian opacity */
  float opacityLearningRate();

  /** @brief Get learning rate for Gaussian scaling */
  float scalingLearningRate();

  /** @brief Get learning rate for Gaussian rotation */
  float rotationLearningRate();

  /** @brief Get SSIM loss weight (lambda_dssim) */
  float lambdaDssim();

  /** @brief Get depth loss weight */
  float lambdaDepth();

  /** @brief Get times of use threshold for new keyframes */
  int newKeyframeTimesOfUse();

  /** @brief Get stability iteration threshold */
  int stableNumIterExistence();

  /** @brief Check if training should continue */
  bool isKeepingTraining();

  // ========== Parameter Setters ==========

  /**
   * @brief Set SSIM loss weight
   * @param lambda_dssim Weight for DSSIM term in loss
   */
  void setLambdaDssim(const float lambda_dssim);

  /**
   * @brief Set usage count for new keyframes
   * @param times Number of times a new keyframe should be used
   */
  void setNewKeyframeTimesOfUse(const int times);

  /**
   * @brief Set stability threshold for Gaussian existence
   * @param niter Number of iterations for stability
   */
  void setStableNumIterExistence(const int niter);

  /**
   * @brief Set whether to keep training after SLAM shutdown
   * @param keep Continue training flag
   */
  void setKeepTraining(const bool keep);

  // ========== Accessors and Configuration ==========

  /**
   * @brief Get reference to Gaussian model parameters
   * @return Reference to model parameters
   */
  GaussianModelParams &getGaussianModelParams() { return this->model_params_; }

  /**
   * @brief Set the sensor type for the mapper
   * @param sensor_type Sensor configuration (MONOCULAR, STEREO, RGBD)
   */
  void setSensorType(SystemSensorType sensor_type) {
    this->sensor_type_ = sensor_type;
  }

  /**
   * @brief Set trajectory viewer for visualization
   * @param viewer Pointer to trajectory viewer instance
   */
  void setTrajectoryViewer(TrajectoryViewer *viewer) {
    trajectory_viewer_ = viewer;
  }

  // ========== Scene Persistence ==========

  /**
   * @brief Increment usage counter for a keyframe
   * @param pkf Keyframe to update
   * @param times Number of times to increment usage counter
   */
  void increaseKeyframeTimesOfUse(std::shared_ptr<GaussianKeyframe> pkf,
                                  int times);

  /**
   * @brief Save complete scene to disk (Gaussians, keyframes, cameras)
   * @param scene_dir Directory to save scene data
   * @return True if save successful, false otherwise
   */
  bool saveScene(std::filesystem::path scene_dir);

  /**
   * @brief Load complete scene from disk
   * @param scene_dir Directory containing scene data
   * @param optional_camera_path Optional path to camera JSON file
   * @return True if load successful, false otherwise
   */
  bool loadScene(std::filesystem::path scene_dir,
                 std::filesystem::path optional_camera_path = "");

  // ========== External Pose Mode (without ORB-SLAM) ==========

  /**
   * @brief Check if current pose/time warrants creating a new keyframe
   * @param current_pose Current camera pose
   * @param current_time Current timestamp
   * @return True if keyframe should be created
   */
  bool isKeyframe(const Sophus::SE3f &current_pose, double current_time);

  /**
   * @brief Process a new keyframe in external pose mode
   * @param rgb_image RGB image
   * @param depth_or_right_image Depth map or right stereo image
   * @param pose Camera pose
   * @param timestamp Image timestamp
   */
  void handleNewKeyframeFromExternal(cv::Mat &rgb_image,
                                     cv::Mat &depth_or_right_image,
                                     const Sophus::SE3f &pose,
                                     const double timestamp);

  /**
   * @brief Handle incoming frame in external mode (checks if keyframe needed)
   * @param rgb_image RGB image
   * @param depth_or_right_image Depth map or right stereo image
   * @param pose Camera pose
   * @param timestamp Image timestamp
   */
  void handleNewFrameExternal(const cv::Mat &rgb_image,
                              const cv::Mat &depth_or_right_image,
                              const Sophus::SE3f &pose,
                              const double timestamp);

  /**
   * @brief Store recent external data for viewer
   * @param rgb_image RGB image to store
   * @param pose Pose to store
   */
  void setRecentExternalData(const cv::Mat &rgb_image,
                             const Sophus::SE3f &pose);

  /**
   * @brief Retrieve recent external data for viewer
   * @return Tuple of (rgb_image, pose)
   */
  std::tuple<const cv::Mat, const Sophus::SE3f> getRecentExternalData();

  /**
   * @brief Main loop for external pose mode
   */
  void run_external_poses();

  /**
   * @brief Check if external data ingestion has stopped
   * @return True if stopped, false otherwise
   */
  volatile bool isExternalDataStopped() {
    return external_data_stopped_.load(std::memory_order_acquire);
  }

  /**
   * @brief Signal that external data ingestion should stop
   */
  volatile void signalExternalDataStopped() {
    std::cout << "External data stopped" << std::endl;
    external_data_stopped_.store(true, std::memory_order_release);
  }

  /**
   * @brief Set callback function to be called on completion
   * @param callback Function to call when mapping finishes
   */
  void setCompletionCallback(std::function<void()> callback);

  // ========== Loop Closure Pause Mechanism ==========

  /**
   * @brief Check if image ingestion should be paused (e.g., during loop
   * closure)
   * @return True if paused, false otherwise
   */
  bool shouldPauseImageIngestion() const {
    return pause_image_ingestion_.load(std::memory_order_acquire);
  }

  /**
   * @brief Block until pause is released
   *
   * Used to halt image ingestion during loop closure optimization.
   */
  void waitWhilePaused() {
    while (pause_image_ingestion_.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  // ========== Public Data Members ==========

  // Configuration
  std::filesystem::path config_file_path_;

  // Core components
  std::shared_ptr<GaussianModel> gaussians_;
  std::shared_ptr<GaussianScene> scene_;
  std::shared_ptr<KeyframeSelection> keyframe_selector_;
  std::shared_ptr<ORB_SLAM3::System> pSLAM_;

  // Chunk management
  float chunk_size_ = 10.0;
  float keyframe_selection_chunk_size_ = 200.0;
  std::filesystem::path chunk_save_dir_;
  std::filesystem::path keyframe_save_dir_;

  // Device and rendering settings
  torch::DeviceType device_type_;
  int num_gaus_pyramid_sub_levels_ = 0;
  std::vector<int> kf_gaus_pyramid_times_of_use_;
  std::vector<float> kf_gaus_pyramid_factors_;

  bool viewer_camera_id_set_ = false;
  std::uint32_t viewer_camera_id_ = 0;
  float rendered_image_viewer_scale_ = 1.0f;
  float rendered_image_viewer_scale_main_ = 1.0f;

  float z_near_ = 0.01f;
  float z_far_ = 100.0f;

  // Undistortion masks
  std::map<camera_id_t, torch::Tensor> undistort_mask_;
  std::map<camera_id_t, torch::Tensor> viewer_main_undistort_mask_;
  std::map<camera_id_t, torch::Tensor> viewer_sub_undistort_mask_;

  // Training tracking
  std::map<std::size_t, float> kfs_loss_;
  std::map<std::size_t, int> kfs_used_times_;

  // Status flags
  bool initial_mapped_;
  bool interrupt_training_;
  bool stopped_;
  int iteration_;
  float ema_loss_for_log_;
  bool SLAM_ended_;
  bool loop_closure_iteration_;
  bool keep_training_ = false;
  int default_sh_ = 0;

  // Sensor configuration
  SystemSensorType sensor_type_;

  // Depth estimation
  float stereo_baseline_length_ = 0.0f;
  cv::Mat stereo_Q_;
  std::shared_ptr<StereoDepth> stereo_depth_estimator_;
  std::shared_ptr<MonoDepth> monocular_depth_estimator_;
  float min_depth_ = 0.0f;
  float max_depth_ = 100.0f;

  // Feature extraction and MVS
  std::unique_ptr<XFeat::XFDetector> feat_extractor_;
  std::unique_ptr<GuidedMVS> guided_mvs_;

  // Mapping parameters
  unsigned long min_num_initial_map_kfs_;
  torch::Tensor background_;
  float large_rot_th_;
  float large_trans_th_;
  torch::Tensor override_color_;

  int new_keyframe_times_of_use_;
  int local_BA_increased_times_of_use_;
  int loop_closure_increased_times_of_use_;

  int stable_num_iter_existence_;

  // Output and recording
  std::filesystem::path result_dir_;
  int keyframe_record_interval_;
  int all_keyframes_record_interval_;
  bool record_rendered_image_;
  bool record_ground_truth_image_;
  bool record_loss_image_;

  int training_report_interval_;
  bool record_loop_ply_;

  // Training metrics
  int metrics_collection_interval_ = 1000;
  struct TrainingMetrics {
    int iteration;
    double elapsed_time_seconds;
    int active_gaussian_count;
    int total_gaussian_count;
    float reserved_memory_mb;
    float allocated_memory_mb;
    float ram_usage_mb;
    int queue_keyframes;
  };
  std::vector<TrainingMetrics> training_metrics_;
  std::chrono::steady_clock::time_point training_start_time_;

  // Sampling parameters
  int exposure_optimization_ = 0;
  float init_proba_scaler_ = 2.0;
  bool downsample_for_sampling_ = false;

  // Utilities
  std::random_device rd_;
  TrajectoryViewer *trajectory_viewer_ = nullptr;

  // External pose mode data
  cv::Mat external_image_;
  Sophus::SE3f external_pose_;
  LeakyFrameQueue frame_queue_;

  std::atomic<bool> external_data_stopped_{false};
  std::function<void()> completion_callback_;

  // Keyframe selection thresholds for external pose mode
  Sophus::SE3f last_keyframe_pose_;
  float min_keyframe_translation_{
      0.25f};  ///< Min translation (meters) for new keyframe
  float min_keyframe_rotation_{
      0.15f};  ///< Min rotation (radians) for new keyframe
  double last_keyframe_timestamp_{0.0};
  float min_keyframe_time_{
      0.5f};  ///< Min time interval (seconds) between keyframes

  // Synchronization
  std::mutex mutex_status_;
  std::mutex mutex_settings_;
  std::mutex mutex_render_;
  std::mutex mutex_external_data_;

 protected:
  // ========== Mapping Conditions and Operations ==========

  /**
   * @brief Check if initial mapping phase can begin
   * @return True if minimum keyframes exist and SLAM has mapping data
   */
  bool hasMetInitialMappingConditions();

  /**
   * @brief Check if incremental mapping can proceed
   * @return True if SLAM has new mapping operations
   */
  bool hasMetIncrementalMappingConditions();

  /**
   * @brief Process queued mapping operations from ORB-SLAM
   *
   * Handles local bundle adjustment results and loop closure operations,
   * updating keyframe poses and Gaussian positions accordingly.
   */
  void combineMappingOperations();

  /**
   * @brief Process a batch of local mapping operations
   * @param operations Vector of mapping operations to process
   */
  void processLocalMappingBABatch(
      std::vector<ORB_SLAM3::MappingOperation> &operations);

  /**
   * @brief Handle loop closure bundle adjustment
   * @param opr Loop closure operation containing updated poses
   */
  void processLoopClosureBA(ORB_SLAM3::MappingOperation &opr);

  /**
   * @brief Process loop closure sequentially for affected keyframes
   * @param associated_kfs Vector of keyframe tuples with updated poses
   * @param loop_kf_scale Scale factor for loop closure keyframes
   * @return Number of gaussians transformed
   */
  int processSequentialLoopClosure(
      const std::vector<KeyframeTuple> &associated_kfs,
      float loop_kf_scale);

  /**
   * @brief Process loop closure with batch loading of chunks
   * @param associated_kfs Vector of keyframe tuples with updated poses
   * @param kf_chunk_pairs Keyframe-chunk pairs to process
   * @param all_unique_chunks Set of unique chunk IDs involved
   * @param loop_kf_scale Scale factor for loop closure keyframes
   * @return Number of gaussians transformed
   */
  int processBatchedLoopClosure(
      std::vector<KeyframeTuple> &associated_kfs,
      const std::vector<std::pair<std::shared_ptr<GaussianKeyframe>,
                                  torch::Tensor>> &kf_chunk_pairs,
      const std::unordered_set<int64_t> &all_unique_chunks,
      float loop_kf_scale);

  /**
   * @brief Refine scene scale based on scale drift correction
   * @param opr Mapping operation containing scale refinement data
   */
  void processScaleRefinement(ORB_SLAM3::MappingOperation &opr);

  /**
   * @brief Check if pose difference exceeds threshold for large correction
   * @param diff_pose Difference between old and new poses
   * @return True if rotation or translation exceeds thresholds
   */
  bool isPoseDivergenceLarge(const Sophus::SE3f &diff_pose) const;

  /**
   * @brief Get relevant chunk IDs visible from a keyframe (loaded, on-disk, or
   * with gaussians)
   * @param visible_chunk_ids Tensor of chunk IDs in keyframe frustum
   * @return Filtered tensor of relevant chunk IDs
   */
  torch::Tensor filterRelevantChunks(
      const torch::Tensor &visible_chunk_ids) const;

  // ========== Keyframe Management ==========

  /**
   * @brief Common keyframe initialization for both ORB-SLAM and external modes
   *
   * Creates a keyframe, estimates depth, samples Gaussians, and adds to scene.
   *
   * @param pkf Output keyframe pointer to initialize
   * @param rgb_image RGB image data
   * @param aux_image Auxiliary image (depth or right stereo image)
   * @param camera Camera parameters
   * @param filename Optional filename for tracking
   */
  void createAndInitializeKeyframe(std::shared_ptr<GaussianKeyframe> &pkf,
                                   cv::Mat &rgb_image,
                                   cv::Mat &aux_image,
                                   const Camera &camera,
                                   const std::string &filename = "",
                                   bool skip_sampling = false);

  /**
   * @brief Handle new keyframe from ORB-SLAM system
   * @param kf Keyframe data tuple from ORB-SLAM
   * @param skip_sampling If true, only create keyframe without sampling
   */
  void handleNewKeyframeFromORBSLAM(KeyframeTuple &kf,
                                    bool skip_sampling = false);

  /**
   * @brief Find N closest keyframes to a given keyframe
   * @param current_kf Reference keyframe
   * @param n Number of closest keyframes to find
   * @param k Stride for keyframe selection (default: 1)
   * @return Vector of closest keyframes
   */
  std::vector<std::shared_ptr<GaussianKeyframe>> getClosestKeyframes(
      std::shared_ptr<GaussianKeyframe> current_kf,
      int n,
      int k = 1);

  // ========== Gaussian Sampling and Recording ==========

  /**
   * @brief Sample new Gaussians from keyframe
   * @param pkf Keyframe to sample Gaussians from
   */
  void sampleGaussians(std::shared_ptr<GaussianKeyframe> pkf);

  /**
   * @brief Record rendered image, ground truth, and loss visualization
   * @param rendered Rendered image tensor
   * @param ground_truth Ground truth image tensor
   * @param kfid Keyframe ID for filename
   * @param result_img_dir Directory for rendered images
   * @param result_gt_dir Directory for ground truth images
   * @param result_loss_dir Directory for loss visualizations
   * @param name_suffix Optional filename suffix
   */
  void recordKeyframeRendered(torch::Tensor &rendered,
                              torch::Tensor &ground_truth,
                              unsigned long kfid,
                              std::filesystem::path result_img_dir,
                              std::filesystem::path result_gt_dir,
                              std::filesystem::path result_loss_dir,
                              std::string name_suffix = "");

  /**
   * @brief Render and record a single keyframe with quality metrics
   * @param pkf Keyframe to render
   * @param dssim Output DSSIM metric
   * @param psnr Output PSNR metric
   * @param psnr_gs Output Gaussian splatting PSNR
   * @param render_time Output rendering time in seconds
   * @param result_img_dir Directory for rendered images
   * @param result_gt_dir Directory for ground truth images
   * @param result_loss_dir Directory for loss visualizations
   * @param name_suffix Optional filename suffix
   */
  void renderAndRecordKeyframe(std::shared_ptr<GaussianKeyframe> pkf,
                               float &dssim,
                               float &psnr,
                               float &psnr_gs,
                               double &render_time,
                               std::filesystem::path result_img_dir,
                               std::filesystem::path result_gt_dir,
                               std::filesystem::path result_loss_dir,
                               std::string name_suffix = "");

  /**
   * @brief Render and record all keyframes in the scene
   * @param name_suffix Optional filename suffix
   */
  void renderAndRecordAllKeyframes(std::string name_suffix = "");

  /**
   * @brief Export keyframe data to JSON format
   * @param result_dir Output directory for JSON file
   */
  void keyframesToJson(std::filesystem::path result_dir);

  /**
   * @brief Write keyframe usage statistics to file
   * @param result_dir Output directory
   * @param name_suffix Optional filename suffix
   */
  void writeKeyframeUsedTimes(std::filesystem::path result_dir,
                              std::string name_suffix = "");

  /**
   * @brief Export training metrics to CSV file
   * @param result_dir Output directory for CSV file
   */
  void writeTrainingMetricsCSV(std::filesystem::path result_dir);

  /**
   * @brief Save chunk manifest file
   * @param scene_dir Scene directory
   */
  void saveChunkManifest(std::filesystem::path scene_dir);

  /**
   * @brief Load chunk manifest file
   * @param scene_dir Scene directory
   */
  void loadChunkManifest(std::filesystem::path scene_dir);

  /**
   * @brief Load camera parameters from JSON file
   * @param json_path Path to cameras JSON file
   */
  void loadCamerasFromJson(std::filesystem::path json_path);

  /**
   * @brief Helper to initialize a camera from intrinsic parameters
   *
   * Common logic for setting up pinhole camera with distortion from intrinsic
   * parameters. Used by both loadScene and loadCamerasFromJson.
   *
   * @param camera_id Camera identifier
   * @param width Image width
   * @param height Image height
   * @param fx Focal length in x direction
   * @param fy Focal length in y direction
   * @param cx Principal point x coordinate
   * @param cy Principal point y coordinate
   * @param k1 Radial distortion coefficient k1
   * @param k2 Radial distortion coefficient k2
   * @param p1 Tangential distortion coefficient p1
   * @param p2 Tangential distortion coefficient p2
   * @param k3 Radial distortion coefficient k3
   */
  void initializeCameraFromIntrinsics(camera_id_t camera_id,
                                      int width,
                                      int height,
                                      float fx,
                                      float fy,
                                      float cx,
                                      float cy,
                                      float k1,
                                      float k2,
                                      float p1,
                                      float p2,
                                      float k3);

  /**
   * @brief Save all Gaussians to PLY file
   * @param name_suffix Filename suffix
   */
  void saveTotalGaussians(std::string name_suffix);

  // ========== Depth Estimation ==========

  // Constants
  static constexpr const char *DEPTH_MODEL_BASE_DIR = "/workspace/repo/models/";
  static constexpr int LOG_KERNEL_RADIUS = 3;
  static constexpr int STEREO_MODEL_HEIGHT = 384;
  static constexpr int STEREO_MODEL_WIDTH = 1280;

  // Depth-related data
  torch::Tensor disc_kernel_;
  float log_sigma_ = 3.0f;

  /**
   * @brief Compute Laplacian of Gaussian (LoG) probability map for edge
   * detection
   *
   * Applies a Laplacian filter followed by smoothing with a disc kernel to
   * detect edges and regions of high spatial variation. Used to identify areas
   * with high information content for point sampling.
   *
   * @param image Input image tensor [C, H, W]
   * @return Probability map tensor [H, W] with values in [0, 1]
   */
  torch::Tensor computeLoGProbability(const torch::Tensor &image);

  /**
   * @brief Initialize circular disc kernel for smoothing LoG responses
   *
   * Creates a normalized disc-shaped convolution kernel used to smooth
   * the Laplacian response and reduce noise in edge detection.
   */
  void initializeLaplacianOfGaussianKernel();

  /**
   * @brief Initialize monocular depth estimator with Depth-Anything v2 model
   *
   * Loads the Depth-Anything v2 ViT-Large model for monocular depth estimation.
   */
  void initializeMonocularDepthEstimator();

  /**
   * @brief Initialize stereo depth estimator with Fast-ACVNet model
   *
   * Loads the Fast-ACVNet+ model for stereo depth estimation at the
   * configured resolution.
   */
  void initializeStereoDepthEstimator();

  /**
   * @brief Extract valid keypoints for depth scale alignment
   *
   * Retrieves ORB-SLAM keypoints and their 3D positions for aligning
   * monocular depth estimates with sparse SLAM reconstruction.
   *
   * @param pkf Keyframe to extract keypoints from
   * @return Tuple of (pixel_coordinates, points_3d_local)
   */
  std::tuple<std::vector<float>, std::vector<float>>
  extractValidKeypointsForDepthAlignment(
      std::shared_ptr<GaussianKeyframe> pkf) const;

  /**
   * @brief Sample confidence values at specified UV coordinates
   *
   * Uses bilinear interpolation to sample from a confidence map at given
   * pixel coordinates.
   *
   * @param mono_depth_conf Confidence map tensor [1, 1, H, W]
   * @param uv Pixel coordinates tensor [N, 2] where N is number of points
   * @param width Image width for coordinate normalization
   * @param height Image height for coordinate normalization
   * @return Sampled confidence values [N]
   */
  torch::Tensor sampleConf(const torch::Tensor &mono_depth_conf,
                           const torch::Tensor &uv,
                           int width,
                           int height);

  // ========== Protected Data Members ==========

  // Model parameters
  GaussianModelParams model_params_;
  GaussianOptimizationParams opt_params_;
  GaussianPipelineParams pipe_params_;

 private:
  // ========== Initialization Helpers ==========

  /**
   * @brief Initialize random seed for reproducibility
   * @param seed Random seed value
   */
  void initializeRandomSeed(int seed);

  /**
   * @brief Initialize compute device (CPU or CUDA)
   * @param device_type Requested device type
   */
  void initializeDevice(torch::DeviceType device_type);

  /**
   * @brief Create output directory structure
   */
  void initializeDirectories();

  /**
   * @brief Initialize background color and override color tensors
   */
  void initializeBackgroundAndOverrideColor();

  /**
   * @brief Initialize core Gaussian components (model, scene, keyframe
   * selector)
   * @param with_training_infrastructure If true, initialize training components
   */
  void initializeGaussianComponents(bool with_training_infrastructure);

  /**
   * @brief Load ORB-SLAM settings and extract camera information
   * @param sensor_type ORB-SLAM sensor type
   * @param orb_settings_path Path to ORB-SLAM settings file
   * @return Tuple of (orb_settings, cameras, image_size, undistort_params)
   */
  std::tuple<ORB_SLAM3::Settings *,
             std::vector<ORB_SLAM3::GeometricCamera *>,
             cv::Size,
             UndistortParams>
  initializeORBSettings(ORB_SLAM3::System::eSensor sensor_type,
                        const std::string &orb_settings_path);

  /**
   * @brief Configure sensor type and stereo parameters
   * @param sensor_type ORB-SLAM sensor type
   * @param orb_settings ORB-SLAM settings object
   */
  void initializeSensorType(ORB_SLAM3::System::eSensor sensor_type,
                            ORB_SLAM3::Settings *orb_settings);

  /**
   * @brief Initialize depth estimation pipelines based on sensor type
   * @return MVS inverse depth range parameter
   */
  float initializeDepthEstimation();

  /**
   * @brief Initialize MVS and feature extraction components
   * @param mvs_inverse_depth_range Inverse depth range for MVS
   */
  void initializeMVSAndFeatures(float mvs_inverse_depth_range);

  /**
   * @brief Process and configure a single camera from ORB-SLAM
   * @param SLAM_camera ORB-SLAM camera object
   * @param undistort_params Undistortion parameters
   * @param SLAM_im_size Image size from SLAM
   */
  void processCameraFromORBSLAM(ORB_SLAM3::GeometricCamera *SLAM_camera,
                                const UndistortParams &undistort_params,
                                const cv::Size &SLAM_im_size);

  // ========== Private Data Members ==========

  // Loop closure control
  std::atomic<bool> pause_image_ingestion_{false};
  int loop_closure_optimization_iterations_ = 1000;
  float loop_closure_memory_multiplier_ =
      8.0f;  ///< Multiplier for max_gaussians_in_memory during loop closure
};

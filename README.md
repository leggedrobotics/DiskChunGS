# DiskChunGS: Large-Scale 3D Gaussian SLAM Through Chunk-Based Memory Management
[Casimir Feldmann](https://scholar.google.com/citations?user=WNqWurwAAAAJ&hl=en&oi=ao)<sup>1</sup>, [Max Wilder-Smith](https://github.com/maxwildersmith)<sup>1</sup>, [Vaishakh Patil](https://scholar.google.com/citations?user=aB04078AAAAJ&hl=en)<sup>1</sup>, [Michael Niemeyer](https://m-niemeyer.github.io/)<sup>2</sup>, [Michael Oechsle](https://moechsle.github.io/)<sup>2</sup>, [Keisuke Tateno](https://scholar.google.com/citations?user=ml3laqEAAAAJ&hl=ja)<sup>2</sup>, and [Marco Hutter](https://scholar.google.ch/citations?user=DO3quJYAAAAJ&hl=en)<sup>1</sup> <br>
ETH Zurich<sup>1</sup>, Google<sup>2</sup>
<br>
[[`Paper`](https://arxiv.org/abs/2511.23030)] [[`Project`](https://rffr.leggedrobotics.com/works/diskchungs/)] [[`Video`](https://www.youtube.com/watch?v=BFqPBZulrhQ&feature=youtu.be)]

![Pipeline](assets/pipeline.png?raw=true)

## Overview

Recent advances in 3D Gaussian Splatting (3DGS) have demonstrated impressive results for novel view synthesis with real-time rendering capabilities. However, integrating 3DGS with SLAM systems faces a fundamental scalability limitation: methods are constrained by GPU memory capacity, restricting reconstruction to small-scale environments. We present DiskChunGS, a scalable 3DGS SLAM system that overcomes this bottleneck through an out-of-core approach that partitions scenes into spatial chunks and maintains only active regions in GPU memory while storing inactive areas on disk. Our architecture integrates seamlessly with existing SLAM frameworks for pose estimation and loop closure, enabling globally consistent reconstruction at scale. We validate DiskChunGS on indoor scenes (Replica, TUM-RGBD), urban driving scenarios (KITTI), and resource-constrained Nvidia Jetson platforms. Our method uniquely completes all 11 KITTI sequences without memory failures while achieving superior visual quality, demonstrating that algorithmic innovation can overcome the memory constraints that have limited previous 3DGS SLAM methods.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Evaluation](#evaluation)
- [ROS Usage](#ros-usage)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Installation

```bash
# Clone this repo
git clone --recursive git@github.com:leggedrobotics/DiskChunGS.git

cd DiskChunGS

# Build the development container (for Jetson use docker-compose_jetson.yml)
docker compose -f docker/docker-compose.yml build dev

# Start the development container (for Jetson use docker-compose_jetson.yml)
docker compose -f docker/docker-compose.yml run --rm dev

# Inside the container, build the application
./scripts/build.sh
```

## Getting Started

### Preparing Datasets

The benchmark datasets mentioned in our paper:
- [Replica (NICE-SLAM Version)](https://github.com/cvg/nice-slam)
- [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)
- [KITTI](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)

1. Create a dataset folder to bind into the container. Edit the dev section of [docker-compose.yml](docker/docker-compose.yml) to set the path:
   ```- /path/to/your/datasets:/data```

2. Download the desired dataset:
   ```bash
   scripts/download_replica.sh
   scripts/download_tum.sh
   ```

   For KITTI:
   - Download odometry data set (color, 65 GB)
   - Download odometry ground truth poses (4 MB)
   - Download odometry data set (calibration files, 1 MB)

   Then combine these to create the following file structure:
   ```
   kitti
   ├── data_odometry_color
       └── dataset
           |── sequences
           |   |── 00
           |   |   |── calib.txt
           |   |   |── image_2
           |   |   |── image_3
           |   |   |── times.txt
           |   |── 01
           |   ...
           └── poses
               |── 00.txt
               |── 01.txt
               ...
   ```

### Running the System

1. Allow the container to connect to your display:
   ```bash
   xhost +local:root
   ```
   Run `xhost -local:root` when done.

2. Run the system with the paths set for your environment. Add `no_viewer` to disable the viewer for evaluation:
   ```bash
   bin/replica_rgbd \
       third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
       cfg/ORB_SLAM3/RGB-D/Replica/office0.yaml \
       cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
       /data/Replica/office0 \
       results/replica_rgbd/office0
       # no_viewer
   ```

3. We also provide scripts to conduct experiments on all benchmark datasets. In case you use a different data location, change the dataset root lines in scripts/*.sh:
   ```bash
   scripts/replica_mono.sh exp_name num_trials
   scripts/replica_rgbd.sh exp_name num_trials
   scripts/tum_mono.sh exp_name num_trials
   scripts/tum_rgbd.sh exp_name num_trials
   scripts/kitti_stereo.sh exp_name num_trials
   # etc.
   ```

## Evaluation

To run evaluation, results must be in the following format. Using the provided `scripts/*.sh` scripts will produce this structure automatically:
```
results
├── replica_mono_0
│   ├── office0
│   ├── ....
│   └── room2
├── replica_rgbd_0
│   ├── office0
│   ├── ....
│   └── room2
│
└── [replica/tum/kitti]_[mono/rgbd/stereo]_num  ....
    ├── scene_1
    ├── ....
    └── scene_n
```

### Enable recording for evaluation

Make sure to enable the following in your gaussian_mapper config before running:
```
Record.record_rendered_image: 1
Record.record_ground_truth_image: 1
```

### Convert Replica GT camera pose files for EVO package:
```bash
cd eval
python3 shapeReplicaGT.py --replica_dataset_path PATH_TO_REPLICA_DATASET
```

### To get all metrics:
```bash
cd eval
python3 eval.py --dataset_center_path PATH_TO_ALL_DATASET --result_main_folder RESULTS_PATH
```

- PATH_TO_ALL_DATASET: Should be /data if you've bound your datasets folder to /data
- Results will be summarized in two files: `RESULTS_PATH/log.txt` and `RESULTS_PATH/log.csv`.


## ROS Usage

### Setup

```bash
docker compose -f docker/docker-compose.yml run --rm dev
source scripts/build_ros.sh
```

### Running the ROS Node

```bash
source /root/catkin_ws/devel/setup.bash
roslaunch diskchungs_ros diskchungs.launch \
vocabulary_path:=/workspace/repo/third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
orb_settings_path:=/workspace/repo/cfg/ORB_SLAM3/RGB-D/RSL/arche_train1.yaml \
gaussian_settings_path:=/workspace/repo/cfg/gaussian_mapper/RGB-D/RSL/arche_train1.yaml \
output_directory:=/workspace/repo/results/rsl/train1 \
use_viewer:=true \
mode:=rgbd \
image_topic:=/left_camera_rgb \
depth_topic:=/zed2/zed_node/depth/depth_registered \
slam_mode:=external \
target_frame:=map \
source_frame:=zed2_left_camera_optical_frame
```

If your camera publishes compressed images, republish as raw first:
```bash
rosrun image_transport republish compressed in:=/zed2/zed_node/left/image_rect_color raw out:=/left_camera_rgb
```

When replaying rosbags, add `--clock` to `roslaunch` and `--pause` to `rosbag play` to ensure TF timestamps align with bag time.

### ROS Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocabulary_path` | string | (required) | Path to the ORB vocabulary file |
| `orb_settings_path` | string | (required) | Path to the ORB SLAM settings file |
| `gaussian_settings_path` | string | (required) | Path to the Gaussian settings file |
| `output_directory` | string | (required) | Directory where the output will be saved |
| `use_viewer` | bool | `false` | Whether to use the ImGui viewer for visualization |
| `mode` | string | `"stereo"` | Sensor mode. Options: `"mono"`, `"stereo"`, `"rgbd"` |
| `slam_mode` | string | `"orbslam"` | SLAM mode. Options: `"orbslam"` (use ORB-SLAM3 for poses), `"external"` (use TF poses; not supported with mono mode) |
| `image_topic` | string | `"/camera/image_raw"` | Primary image topic: left camera (stereo), mono image (mono), or RGB image (rgbd) |
| `right_topic` | string | `"/camera/right/image_raw"` | Right camera topic (stereo mode only) |
| `depth_topic` | string | `"/camera/depth/image_raw"` | Depth image topic (rgbd mode only) |
| `target_frame` | string | `"map"` | Target frame for TF transformations (external mode) |
| `source_frame` | string | `"zed2i_left_camera_frame"` | Source frame for TF transformations (external mode) |
| `timeout_duration` | double | `20.0` | Duration (in seconds) after which the system considers the data stream stopped |

When using external mode, configure keyframe selection thresholds in your gaussian_mapper config:
```
External.min_keyframe_translation: 0.25
External.min_keyframe_rotation: 0.15
External.min_keyframe_time: 0.5
```

## Notes

**Jetson Usage:**
In `cfg/gaussian_mapper/Stereo/KITTI/KITTI.yaml`, reduce `sh_degree` to `2`:
```yaml
Model.sh_degree: 2
```
The ROS wrapper is not supported in the current Jetson version of DiskChunGS.

## License

DiskChunGS is licensed under the **GNU General Public License v3.0** (GPL v3).

The `third_party/gaussian_splatting/` directory contains code derived from Inria
research projects (3D Gaussian Splatting, On-The-Fly-NVS) which are licensed
under Inria's non-commercial research license. Commercial use of these
components requires a separate license from Inria (stip-sophia.transfert@inria.fr).

See [LICENSE.md](LICENSE.md) for full details including all upstream licenses.

## Acknowledgements

This work incorporates many open-source codes. Thanks for their great work!
- [CaRtGS](https://github.com/DapengFeng/cartgs)
- [Photo-SLAM](https://github.com/HuajianUP/Photo-SLAM)
- [Taming 3DGS](https://github.com/humansensinglab/taming-3dgs)
- [Frustum Culling](https://bruop.github.io/frustum_culling/)
- [On-the-fly](https://repo-sam.inria.fr/nerphys/on-the-fly-nvs/)
- [depth-anything-tensorrt](https://github.com/spacewalk01/depth-anything-tensorrt)
- [xfeat_cpp](https://github.com/udaysankar01/xfeat_cpp)
- [ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)

## Citation

If you find this work useful in your research, consider citing it:
```
@ARTICLE{11417439,
  author={Feldmann, Casimir and Wilder-Smith, Maximum and Patil, Vaishakh and Oechsle, Michael and Niemeyer, Michael and Tateno, Keisuke and Hutter, Marco},
  journal={IEEE Robotics and Automation Letters}, 
  title={DiskChunGS: Large-Scale 3D Gaussian SLAM Through Chunk-Based Memory Management}, 
  year={2026},
  volume={11},
  number={4},
  pages={5009-5016},
  keywords={Simultaneous localization and mapping;Three-dimensional displays;Rendering (computer graphics);Random access memory;Loss measurement;Visualization;Optimization;Memory management;Graphics processing units;Real-time systems;Large-scale reconstruction;mapping;3D Gaussian Splatting;slam},
  doi={10.1109/LRA.2026.3668704}}
```
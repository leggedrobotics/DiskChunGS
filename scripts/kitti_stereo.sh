#!/bin/bash
exp=$1
num_trials=$2

for ((i=0; i<num_trials; i++))
do
bin/kitti_stereo \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/Stereo/KITTI/KITTI00-02.yaml \
    cfg/gaussian_mapper/Stereo/KITTI/KITTI.yaml \
    /data/kitti/data_odometry_color/dataset/sequences/00 \
    results/$exp/kitti_stereo_$i/00 \
    no_viewer \
    5.0 \
    true
done

for ((i=0; i<num_trials; i++))
do
bin/kitti_stereo \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/Stereo/KITTI/KITTI00-02.yaml \
    cfg/gaussian_mapper/Stereo/KITTI/KITTI.yaml \
    /data/kitti/data_odometry_color/dataset/sequences/01 \
    results/$exp/kitti_stereo_$i/01 \
    no_viewer \
    5.0 \
    true
done

for ((i=0; i<num_trials; i++))
do
bin/kitti_stereo \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/Stereo/KITTI/KITTI00-02.yaml \
    cfg/gaussian_mapper/Stereo/KITTI/KITTI.yaml \
    /data/kitti/data_odometry_color/dataset/sequences/02 \
    results/$exp/kitti_stereo_$i/02 \
    no_viewer \
    5.0 \
    true
done

for ((i=0; i<num_trials; i++))
do
bin/kitti_stereo \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/Stereo/KITTI/KITTI03.yaml \
    cfg/gaussian_mapper/Stereo/KITTI/KITTI.yaml \
    /data/kitti/data_odometry_color/dataset/sequences/03 \
    results/$exp/kitti_stereo_$i/03 \
    no_viewer \
    5.0 \
    true
done

for ((i=0; i<num_trials; i++))
do
bin/kitti_stereo \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/Stereo/KITTI/KITTI04-12.yaml \
    cfg/gaussian_mapper/Stereo/KITTI/KITTI.yaml \
    /data/kitti/data_odometry_color/dataset/sequences/04 \
    results/$exp/kitti_stereo_$i/04 \
    no_viewer \
    5.0 \
    true
done

for ((i=0; i<num_trials; i++))
do
bin/kitti_stereo \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/Stereo/KITTI/KITTI04-12.yaml \
    cfg/gaussian_mapper/Stereo/KITTI/KITTI.yaml \
    /data/kitti/data_odometry_color/dataset/sequences/05 \
    results/$exp/kitti_stereo_$i/05 \
    no_viewer \
    5.0 \
    true
done

for ((i=0; i<num_trials; i++))
do
bin/kitti_stereo \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/Stereo/KITTI/KITTI04-12.yaml \
    cfg/gaussian_mapper/Stereo/KITTI/KITTI.yaml \
    /data/kitti/data_odometry_color/dataset/sequences/06 \
    results/$exp/kitti_stereo_$i/06 \
    no_viewer \
    5.0 \
    true
done

for ((i=0; i<num_trials; i++))
do
bin/kitti_stereo \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/Stereo/KITTI/KITTI04-12.yaml \
    cfg/gaussian_mapper/Stereo/KITTI/KITTI.yaml \
    /data/kitti/data_odometry_color/dataset/sequences/07 \
    results/$exp/kitti_stereo_$i/07 \
    no_viewer \
    5.0 \
    true
done

for ((i=0; i<num_trials; i++))
do
bin/kitti_stereo \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/Stereo/KITTI/KITTI04-12.yaml \
    cfg/gaussian_mapper/Stereo/KITTI/KITTI.yaml \
    /data/kitti/data_odometry_color/dataset/sequences/08 \
    results/$exp/kitti_stereo_$i/08 \
    no_viewer \
    5.0 \
    true
done

for ((i=0; i<num_trials; i++))
do
bin/kitti_stereo \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/Stereo/KITTI/KITTI04-12.yaml \
    cfg/gaussian_mapper/Stereo/KITTI/KITTI.yaml \
    /data/kitti/data_odometry_color/dataset/sequences/09 \
    results/$exp/kitti_stereo_$i/09 \
    no_viewer \
    5.0 \
    true
done

for ((i=0; i<num_trials; i++))
do
bin/kitti_stereo \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/Stereo/KITTI/KITTI04-12.yaml \
    cfg/gaussian_mapper/Stereo/KITTI/KITTI.yaml \
    /data/kitti/data_odometry_color/dataset/sequences/10 \
    results/$exp/kitti_stereo_$i/10 \
    no_viewer \
    5.0 \
    true
done
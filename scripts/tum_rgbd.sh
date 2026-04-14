#!/bin/bash
exp=$1
num_trials=$2

for ((i=0; i<num_trials; i++))
do
bin/tum_rgbd \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_desk.yaml \
    cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml \
    /data/TUM/rgbd_dataset_freiburg1_desk \
    cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg1_desk.txt \
    results/$exp/tum_rgbd_$i/rgbd_dataset_freiburg1_desk \
    no_viewer
done

for ((i=0; i<num_trials; i++))
do
bin/tum_rgbd \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg2_xyz.yaml \
    cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml \
    /data/TUM/rgbd_dataset_freiburg2_xyz \
    cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg2_xyz.txt \
    results/$exp/tum_rgbd_$i/rgbd_dataset_freiburg2_xyz \
    no_viewer
done

for ((i=0; i<num_trials; i++))
do
bin/tum_rgbd \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
    cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml \
    /data/TUM/rgbd_dataset_freiburg3_long_office_household \
    cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt \
    results/$exp/tum_rgbd_$i/rgbd_dataset_freiburg3_long_office_household \
    no_viewer
done

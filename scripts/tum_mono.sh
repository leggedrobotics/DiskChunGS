#!/bin/bash
exp=$1
num_trials=$2

for ((i=0; i<num_trials; i++))
do
bin/tum_mono \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg1_desk.yaml \
    cfg/gaussian_mapper/Monocular/TUM/tum_mono.yaml \
    /data/TUM/rgbd_dataset_freiburg1_desk \
    results/$exp/tum_mono_$i/rgbd_dataset_freiburg1_desk \
    no_viewer
done

for ((i=0; i<num_trials; i++))
do
bin/tum_mono \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg2_xyz.yaml \
    cfg/gaussian_mapper/Monocular/TUM/tum_mono.yaml \
    /data/TUM/rgbd_dataset_freiburg2_xyz \
    results/$exp/tum_mono_$i/rgbd_dataset_freiburg2_xyz \
    no_viewer
done

for ((i=0; i<num_trials; i++))
do
bin/tum_mono \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg3_long_office_household.yaml \
    cfg/gaussian_mapper/Monocular/TUM/tum_mono.yaml \
    /data/TUM/rgbd_dataset_freiburg3_long_office_household \
    results/$exp/tum_mono_$i/rgbd_dataset_freiburg3_long_office_household \
    no_viewer
done

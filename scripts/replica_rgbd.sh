#!/bin/bash
exp=$1
num_trials=$2

for ((i=0; i<num_trials; i++))
do
bin/replica_rgbd \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/RGB-D/Replica/office0.yaml \
    cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /data/Replica/office0 \
    results/$exp/replica_rgbd_$i/office0 \
    no_viewer
done

for ((i=0; i<num_trials; i++))
do
bin/replica_rgbd \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/RGB-D/Replica/office1.yaml \
    cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /data/Replica/office1 \
    results/$exp/replica_rgbd_$i/office1 \
    no_viewer
done

for ((i=0; i<num_trials; i++))
do
bin/replica_rgbd \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/RGB-D/Replica/office2.yaml \
    cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /data/Replica/office2 \
    results/$exp/replica_rgbd_$i/office2 \
    no_viewer
done

for ((i=0; i<num_trials; i++))
do
bin/replica_rgbd \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/RGB-D/Replica/office3.yaml \
    cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /data/Replica/office3 \
    results/$exp/replica_rgbd_$i/office3 \
    no_viewer
done

for ((i=0; i<num_trials; i++))
do
bin/replica_rgbd \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/RGB-D/Replica/office4.yaml \
    cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /data/Replica/office4 \
    results/$exp/replica_rgbd_$i/office4 \
    no_viewer
done

for ((i=0; i<num_trials; i++))
do
bin/replica_rgbd \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/RGB-D/Replica/room0.yaml \
    cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /data/Replica/room0 \
    results/$exp/replica_rgbd_$i/room0 \
    no_viewer
done

for ((i=0; i<num_trials; i++))
do
bin/replica_rgbd \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/RGB-D/Replica/room1.yaml \
    cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /data/Replica/room1 \
    results/$exp/replica_rgbd_$i/room1 \
    no_viewer
done

for ((i=0; i<num_trials; i++))
do
bin/replica_rgbd \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/RGB-D/Replica/room2.yaml \
    cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /data/Replica/room2 \
    results/$exp/replica_rgbd_$i/room2 \
    no_viewer
done

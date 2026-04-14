#!/bin/bash
exp=$1
num_trials=$2

scripts/kitti_stereo.sh $exp $num_trials
scripts/replica_mono.sh $exp $num_trials
scripts/tum_mono.sh $exp $num_trials
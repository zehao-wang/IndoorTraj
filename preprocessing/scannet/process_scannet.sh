#!/bin/bash

# Check if scene_id is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <scene_id> (e.g., 0134_02)"
  exit 1
fi

dset=scene$1
dataset_dir='../../data/raw/ScanNet'

python process_scannet.py \
--filename ${dataset_dir}/scans/${dset}/${dset}.sens \
--point_clouds_dir ${dataset_dir}/scans/${dset}/${dset}_vh_clean_2.ply \
--output_path ${dataset_dir}/${dset}/ \
--export_color_images --export_poses --export_intrinsics

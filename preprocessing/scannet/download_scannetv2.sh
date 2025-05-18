#!/bin/bash

# Check if scene_id is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <scene_id> (e.g., 0134_02)"
  exit 1
fi

scene_id=$1
dataset_dir='../../data/raw/ScanNet' # Adjust the path to base workspace data folder
python download-scannetv2.py -o ${dataset_dir} --id scene${scene_id} --type .sens
python download-scannetv2.py -o ${dataset_dir} --id scene${scene_id} --type _vh_clean_2.ply

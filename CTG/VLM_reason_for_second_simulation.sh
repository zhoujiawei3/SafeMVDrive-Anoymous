#!/bin/bash

JSON_FILE="VLM_attack_propose/annotation/mini-data_new_500_continue_val_random3_bug_fix_before20frames_VLM_no_type_inference_trajectory.json"
CACHE_DIR="VLM_attack_propose/annotation/cache_5_2_500_val_random3_bug_fix_before20frames_second_time_simulation"
OUTPUT_DIR="VLM_attack_propose/annotation/output_5_2_500_val_random3_bug_fix_before20frames_second_time_simulation"
RESULTS_DIR="CTG/5_2_add_trajectory_val_500_collideloss_intime_remove_second_time_simulation_decay_rate_0.9_buffer_dist7e-1_num_disk_5/"
DATASET_PATH="nuscenes"
POLICY_CKPT_DIR="finetune/CTG_based_current_mask_false/run0/checkpoints"
POLICY_CKPT_KEY="iter80000.ckpt"
refix="500_val_random3_bug_fix_before20frames_second_time_simulation"
trajrefix="second_time_drate_0.9_bdist7e-1_ndisk_5_scope_52_pred"
NUM_GPUS=3  
export CUDA_VISIBLE_DEVICES=4,5,6,7
mkdir -p $CACHE_DIR
mkdir -p $OUTPUT_DIR

python3 << EOF
import json
import os
import math
from collections import defaultdict


with open("$JSON_FILE", "r") as f:
    data = json.load(f)

num_parts = $NUM_GPUS
parts = [[] for _ in range(num_parts)]

for i, item in enumerate(data):
    parts[i % num_parts].append(item)

for i, part in enumerate(parts):
    part_file = os.path.join("$CACHE_DIR", f"part_{i}.json")
    with open(part_file, "w") as f:
        json.dump(part, f, indent=2)
    print(f" {i} {part_file}")


EOF


for i in $(seq 0 $(($NUM_GPUS-1))); do
    PART_FILE="$CACHE_DIR/part_$i.json"
    LOG_FILE="$OUTPUT_DIR/log_gpu_$i.txt"
    


    nohup python scripts/scene_editor_json_input_second_time_simulation.py \
        --results_root_dir $RESULTS_DIR \
        --num_scenes_per_batch 1 \
        --dataset_path $DATASET_PATH \
        --env trajdata \
        --policy_ckpt_dir $POLICY_CKPT_DIR \
        --policy_ckpt_key $POLICY_CKPT_KEY \
        --eval_class Diffuser \
        --editing_source 'config' 'heuristic' \
        --registered_name 'trajdata_nusc_diff_based_current_false' \
        --render \
        --simulation_json_path $PART_FILE \
        --trajdata_source_test val \
        --refix $refix \
        --trajectory_output \
        --trajrefix $trajrefix \
        --cuda $i >> $LOG_FILE 2>&1 &
    


    sleep 2
done



wait



python3 << EOF
import json
import os
import glob


processed_files = []
for i in range($NUM_GPUS):
    part_file = os.path.join("$CACHE_DIR", f"part_{i}.json")
    base_name, ext = os.path.splitext(os.path.basename(part_file))
    processed_file = os.path.join(os.path.dirname(part_file), f"{base_name}_$refix{ext}")
    if os.path.exists(processed_file):
        processed_files.append(processed_file)
    else:
        print(f" {processed_file}")


merged_data = []
for file_path in processed_files:
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                merged_data.extend(data)
            else:
                print(f"{file_path} ")
    except Exception as e:
        print(f" {file_path}  {e}")

original_base_name, original_ext = os.path.splitext(os.path.basename("$JSON_FILE"))
merged_file_path = os.path.join(os.path.dirname("$JSON_FILE"), f"{original_base_name}_$refix{original_ext}")

with open(merged_file_path, "w") as f:
    json.dump(merged_data, f, indent=2)

print(f"{len(merged_data) {merged_file_path}")
EOF

echo "$(dirname $JSON_FILE)/$(basename $JSON_FILE .json)_$refix.json"

echo "-"

python3 << EOF
import json
import os
import glob


processed_files = []
for i in range($NUM_GPUS):
    part_file = os.path.join("$CACHE_DIR", f"part_{i}.json")
    base_name, ext = os.path.splitext(os.path.basename(part_file))
    processed_file = os.path.join(os.path.dirname(part_file), f"{base_name}_$trajrefix{ext}")
    if os.path.exists(processed_file):
        processed_files.append(processed_file)
    else:
        print(f" {processed_file}")


merged_data = []
for file_path in processed_files:
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                merged_data.extend(data)
            else:
                print(f"{file_path} ")
    except Exception as e:
        print(f"{file_path}{e}")


original_base_name, original_ext = os.path.splitext(os.path.basename("$JSON_FILE"))
merged_file_path = os.path.join(os.path.dirname("$JSON_FILE"), f"{original_base_name}_$trajrefix{original_ext}")

with open(merged_file_path, "w") as f:
    json.dump(merged_data, f, indent=2)

print(f" {len(merged_data)} {merged_file_path}")
EOF

echo "$(dirname $JSON_FILE)/$(basename $JSON_FILE .json)_$trajrefix.json"
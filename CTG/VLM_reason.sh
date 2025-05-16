#!/bin/bash

# 
JSON_FILE="VLM_attack_propose/annotation/mini-data_250_val_random3_bug_fix_before_20_frames.json"
CACHE_DIR="VLM_attack_propose/annotation/cache_4_29_250_val_random3_bug_fix_before20frames_CTG_closest_GPTloss"
OUTPUT_DIR="VLM_attack_propose/annotation/output_4_29_250_val_random3_bug_fix_before20frames_CTG_closest_GPTloss"
RESULTS_DIR="CTG/4_29_250_val_collide_random3_bug_fix_before20frames_CTG_closest_GPTloss/"
DATASET_PATH="nuscenes"
POLICY_CKPT_DIR="finetune/CTG_based_current_mask_false/run0/checkpoints"
POLICY_CKPT_KEY="iter80000.ckpt"
refix="250_CTG_closest_GPTloss_simulation"
NUM_GPUS=3  

# 
mkdir -p $CACHE_DIR
mkdir -p $OUTPUT_DIR


python3 << EOF
import json
import os
import math
from collections import defaultdict

# JSON
with open("$JSON_FILE", "r") as f:
    data = json.load(f)
# 
num_parts = $NUM_GPUS
parts = [[] for _ in range(num_parts)]
#parts
for i, item in enumerate(data):
    parts[i % num_parts].append(item)
# 
for i, part in enumerate(parts):
    part_file = os.path.join("$CACHE_DIR", f"part_{i}.json")
    with open(part_file, "w") as f:
        json.dump(part, f, indent=2)
    print(f" {i}  {part_file}")


EOF

# GPU
for i in $(seq 0 $(($NUM_GPUS-1))); do
    PART_FILE="$CACHE_DIR/part_$i.json"
    LOG_FILE="$OUTPUT_DIR/log_gpu_$i.txt"
    
    echo " GPU $i  $PART_FILE"
    
    nohup python scripts/scene_editor_json_input_test_crash_or_VLM_reason.py \
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
        --cuda $i > $LOG_FILE 2>&1 &
    
    echo " GPU $i: $LOG_FILE"
    
    # 
    sleep 2
done

echo "..."

# 
wait

echo "."

# JSON
python3 << EOF
import json
import os
import glob

# JSON
processed_files = []
for i in range($NUM_GPUS):
    part_file = os.path.join("$CACHE_DIR", f"part_{i}.json")
    base_name, ext = os.path.splitext(os.path.basename(part_file))
    processed_file = os.path.join(os.path.dirname(part_file), f"{base_name}_$refix{ext}")
    if os.path.exists(processed_file):
        processed_files.append(processed_file)
    else:
        print(f":  {processed_file}")

# JSON
merged_data = []
for file_path in processed_files:
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                merged_data.extend(data)
            else:
                print(f": {file_path} ")
    except Exception as e:
        print(f" {file_path} : {e}")

# 
original_base_name, original_ext = os.path.splitext(os.path.basename("$JSON_FILE"))
merged_file_path = os.path.join(os.path.dirname("$JSON_FILE"), f"{original_base_name}_$refix{original_ext}")

with open(merged_file_path, "w") as f:
    json.dump(merged_data, f, indent=2)

print(f"!  {len(merged_data)}  {merged_file_path}")
EOF

echo " $(dirname $JSON_FILE)/$(basename $JSON_FILE .json)_$refix.json"

echo ""

python3 << EOF
import json
import os
import glob

# JSON
processed_files = []
for i in range($NUM_GPUS):
    part_file = os.path.join("$CACHE_DIR", f"part_{i}.json")
    base_name, ext = os.path.splitext(os.path.basename(part_file))
    processed_file = os.path.join(os.path.dirname(part_file), f"{base_name}_$refix-trajectory{ext}")
    if os.path.exists(processed_file):
        processed_files.append(processed_file)
    else:
        print(f":  {processed_file}")

# JSON
merged_data = []
for file_path in processed_files:
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                merged_data.extend(data)
            else:
                print(f": {file_path} ")
    except Exception as e:
        print(f" {file_path} : {e}")

# 
original_base_name, original_ext = os.path.splitext(os.path.basename("$JSON_FILE"))
merged_file_path = os.path.join(os.path.dirname("$JSON_FILE"), f"{original_base_name}_$refix-trajectory{original_ext}")

with open(merged_file_path, "w") as f:
    json.dump(merged_data, f, indent=2)

print(f"!  {len(merged_data)}  {merged_file_path}")
EOF

echo " $(dirname $JSON_FILE)/$(basename $JSON_FILE .json)_$refix-trajectory.json"
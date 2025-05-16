#!/bin/bash

# Paths
INPUT_DIR=$1
CONFIG_PATH="/home//OpenDWM/examples/ctsd_unimlvg_6views_video_generation.json"
OUTPUT_BASE_DIR=$2
GPU_ID=$3

# Generate a unique identifier for this script instance
SCRIPT_ID=$(date +%s%N)

# Create a unique temp config file path
TEMP_CONFIG="/tmp/temp_config_${SCRIPT_ID}.json"

# Create output base directory if it doesn't exist
mkdir -p "$OUTPUT_BASE_DIR"

# Save original config at the beginning
cp "$CONFIG_PATH" "/tmp/original_config_${SCRIPT_ID}.json"

echo "Starting processing with script instance ID: ${SCRIPT_ID}"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_BASE_DIR"
echo "Using GPU ID: $GPU_ID"

# Process each directory
for folder in "$INPUT_DIR"/*; do
    if [ ! -d "$folder" ]; then
        continue
    fi
    
    folder_name=$(basename "$folder")
    data_json_path="$folder/data.json"
    
    # Check if data.json exists
    if [ ! -f "$data_json_path" ]; then
        echo "Warning: data.json not found in $folder, skipping..."
        continue
    fi
    
    echo "Processing folder: $folder_name"
    
    # Create output directory for this folder
    output_dir="$OUTPUT_BASE_DIR/$folder_name"
    
    # Check if output directory exists
    if [ -d "$output_dir" ]; then
        echo "Output directory already exists: $output_dir, skipping..."
        continue
    fi
    
    # Create a new temp config file for each iteration based on the original
    cp "/tmp/original_config_${SCRIPT_ID}.json" "$TEMP_CONFIG"
    
    # Update the json_file path in the config
    jq --arg path "$data_json_path" '.validation_dataset.base_dataset.json_file = $path' "/tmp/original_config_${SCRIPT_ID}.json" > "$TEMP_CONFIG"
    
    # Run the command
    echo "Running processing command..."
    cd /home//OpenDWM && \
    CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH=src python src/dwm/preview.py -c "$TEMP_CONFIG" -o "$output_dir"
    
    echo "Finished processing $folder_name"
done

# Clean up temp files
rm "/tmp/original_config_${SCRIPT_ID}.json"
rm "$TEMP_CONFIG"

echo "All folders processed"
import os
import shutil
import random
from collections import defaultdict
random.seed(10)
# 
base_dir = "/data3//OpenDWM"
from_dir= base_dir+"/OpenDWM_test_dataset_convert_video"
subdirs = [
    "5_2_adversarial_trajectory_250_val_41_items_first_stage",
    "4_30_adversarial_trajectory_gen_41_itmes",
    "4_30_origin_trajectory_gen_41_itmes",
]
full_paths = [os.path.join(from_dir, sd) for sd in subdirs]
target_root = os.path.join(base_dir, "OpenDWM_test_dataset_convert_video_pick_test_5")

# scene+frameIndex
file_keys = defaultdict(set)
file_paths = defaultdict(dict)  # 

# 
for subdir, full_path in zip(subdirs, full_paths):
    for fname in os.listdir(full_path):
        if fname.endswith(".mp4"):
            #  scene  startFrameIndex
            parts = fname.split("_")
            scene = next((p for p in parts if p.startswith("scene-")), None)
            frame = next((p for p in parts if p.startswith("startFrameIndex")), None)
            if scene and frame:
                key = f"{scene}_{frame}"
                file_keys[subdir].add(key)
                file_paths[subdir][key] = os.path.join(full_path, fname)

# 
common_keys = set.intersection(*(file_keys[subdir] for subdir in subdirs))
print(f" {len(common_keys)} ")

# 10
selected_keys = random.sample(list(common_keys), min(10, len(common_keys)))
print(f" {len(selected_keys)} ")

# 
for subdir in subdirs:
    target_subdir = os.path.join(target_root, subdir)
    os.makedirs(target_subdir, exist_ok=True)
    for key in selected_keys:
        src = file_paths[subdir][key]
        dst = os.path.join(target_subdir, os.path.basename(src))
        shutil.copy2(src, dst)
        print(f"{src} -> {dst}")
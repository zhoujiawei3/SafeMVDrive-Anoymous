import json
json_path="/home//VLM_attack_propose/annotation/mini-data_20_val_random3_bug_fix_before_20_frames_trajectory_trajectory_20_val_random3_second_time_decay_rate_1_buffer_dist7e-1_num_disk_5_scope_10_GT_only_no_collide.json"
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
total_distance=0.0
for item in data:
    distance = item["min_distance"]
    total_distance+=distance
avg_distance= total_distance/float(len(data))
print("escape_scene_number:",len(data))
print("avg_distance:",avg_distance)

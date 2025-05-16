import json
input_path = "/home//VLM_attack_propose/annotation/mini-data_new_1500_bev_continue_train_after6frames_random7_only_no_type.json"
output_path = "/home//VLM_attack_propose/annotation/mini-data_new_1500_bev_continue_train_after6frames_random7_only_no_type_with_barrier.json"
with open(input_path, "r") as f:
    data = json.load(f)
name_list=['nuscenes_trainval_1.0_719_14.jpg','nuscenes_trainval_1.0_270_19.jpg','nuscenes_trainval_1.0_818_20.jpg','nuscenes_trainval_1.0_453_37.jpg','nuscenes_trainval_1.0_370_21.jpg','nuscenes_trainval_1.0_42_28.jpg','nuscenes_trainval_1.0_369_17.jpg','nuscenes_trainval_1.0_235_31.jpg','nuscenes_trainval_1.0_109_37.jpg']
new_json=[]
for item in data:
    if item['image'] in name_list:
        new_json.append(item)
        
with open(output_path, "w") as f:
    json.dump(new_json, f, indent=4)

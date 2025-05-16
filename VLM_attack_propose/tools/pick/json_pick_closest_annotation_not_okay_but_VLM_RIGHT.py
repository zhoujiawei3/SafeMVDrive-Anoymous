import json

# 
input_path = "/home//VLM_attack_propose/VLM-R1/src/eval/logs_5_9/rgb_results_250_val_random3_bug_fix_after6frames_1500_continue_train_after6frames_random7_bug_fix_audo_label_false_current_speed_add_ego_V_add_noType_collide_question_only_no_type-consine_7B_lora_64_128_0.05_lr_2e-5_deepseed_3/step_2600.json"
output_path = "/home//VLM_attack_propose/VLM-R1/src/eval/logs_5_9/rgb_results_250_val_random3_bug_fix_after6frames_1500_continue_train_after6frames_random7_bug_fix_audo_label_false_current_speed_add_ego_V_add_noType_collide_question_only_no_type-consine_7B_lora_64_128_0.05_lr_2e-5_deepseed_3/step_2600_pick_out_VLM_RIGHT_but_cloest_wrong.json"

#  JSON 
with open(input_path, "r") as f:
    data = json.load(f)

# 
filtered_results = []
for item in data["results"]:
    ground_truth_keys = set(item["ground_truth"].keys())
    model_answer = item["model_answer"]
    
    # 
    # 1. '1'  ground_truth  keys 
    # 2. model_answer  ground_truth  key
    if "1" not in ground_truth_keys and model_answer in ground_truth_keys:
        filtered_results.append(item)

#  JSON 
with open(output_path, "w") as f:
    json.dump({"results": filtered_results}, f, indent=2)

print(f" {len(filtered_results)}  {output_path}")
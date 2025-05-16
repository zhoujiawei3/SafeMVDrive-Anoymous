import json
with open('/home//VLM_attack_propose/annotation/mini-data_new_500_new_train_after6frames_random7_bug_fix_audo_label_false_current_speed_add_ego_V_add_noType_collide_question_1to1_in_each_question_but_notype_all.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print(len(data))
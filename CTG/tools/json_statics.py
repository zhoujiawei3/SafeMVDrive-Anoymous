import json
with open('/home//VLM_attack_propose/annotation/mini-data_1500_continue_train_after6frames_random7_bug_fix_auto_label_test_false_current_speed.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
collisions = {"A vehicle cuts in and collides with the ego vehicle":0,
              "A vehicle rear-ends the ego vehicle":0,
              "Ego vehicle rear-ends another vehicle":0,
              "A vehicle has a head-on collision with the ego vehicle":0,
              "A vehicle has a T-bone collision with the ego vehicle":0}
for item in data:
    collision_type = item['collision']
    reward = item['reward']
    for key in reward.keys():
        if reward[key] == 1:
            collisions[collision_type] += 1
            break
count_collide=0
for item in data:
    collision_dict = item['collision_dict']
    for key in collision_dict.values():
        if key== 1:
            count_collide += 1
            break
print(collisions)
print(count_collide/5)
            
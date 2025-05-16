import json
import random
with open("/home//VLM_attack_propose/annotation/mini-data_new_1500_continue_train_after6frames_random7_bug_fix_audo_label_false_current_speed_add_ego_V_add_noType_collide_question_adjustPrompt.json", "r") as f:
    datas = json.load(f)

desired_number=5

collision_type_list=['A vehicle cuts in and collides with the ego vehicle','A vehicle rear-ends the ego vehicle','Ego vehicle rear-ends another vehicle','A vehicle has a head-on collision with the ego vehicle','A vehicle has a T-bone collision with the ego vehicle']
#
dict_has_car_in_groundtruth_of_different_collision_type={collision_type:[] for collision_type in collision_type_list}
dict_no_car_in_groundtruth_of_different_collision_type={collision_type:[] for collision_type in collision_type_list}
count_of_different_collision_type={collision_type:0 for collision_type in collision_type_list}

dict_hasGT_car_No_type_list=[]
dict_noGT_car_No_type_list=[]
for data in datas:
    if "The desired generated collision type is" in data["conversations"][0]["value"]:
        has_car_in_groundtruth=True
        if not data["conversations"][1]["value"]:
            has_car_in_groundtruth=False
        collision_type=data["conversations"][0]["value"].split("The desired generated collision type is ")[1].split(".")[0]
        if has_car_in_groundtruth:
            dict_has_car_in_groundtruth_of_different_collision_type[collision_type].append(data)
        else:
            dict_no_car_in_groundtruth_of_different_collision_type[collision_type].append(data)
    else:
        has_car_in_groundtruth=True
        if not data["conversations"][1]["value"]:
            has_car_in_groundtruth=False
        if has_car_in_groundtruth:
            dict_hasGT_car_No_type_list.append(data)
        else:
            dict_noGT_car_No_type_list.append(data)
has_car_number=0
new_datas=[]

for collision_type in collision_type_list:
    # 
    random.shuffle(dict_has_car_in_groundtruth_of_different_collision_type[collision_type])
    random.shuffle(dict_no_car_in_groundtruth_of_different_collision_type[collision_type])
    count_of_different_collision_type[collision_type]=len(dict_has_car_in_groundtruth_of_different_collision_type[collision_type][:desired_number])
    # 
    has_samples = dict_has_car_in_groundtruth_of_different_collision_type[collision_type][:desired_number]
    no_samples = dict_no_car_in_groundtruth_of_different_collision_type[collision_type][:desired_number]
    
    # has_sampleno_sample
    for has_data, no_data in zip(has_samples, no_samples):
        new_datas.append(has_data)
        new_datas.append(no_data)






#5collision_type

# count=0
# for i in range(has_car_number//5):
#     #collision_type
#     collision_type=collision_type_list[i%5]
#     data=random.choice(dict_no_car_in_groundtruth_of_different_collision_type[collision_type])
#     new_datas.append(data)
#     count+=1


# print(count)
print(count_of_different_collision_type)

#shuffle new_datas
random.shuffle(new_datas)
with open("/home//VLM_attack_propose/annotation/mini-data_new_1500_continue_train_after6frames_random7_bug_fix_audo_label_false_current_speed_add_ego_V_add_noType_collide_question_all_type_same=5.json", "w") as f:
    json.dump(new_datas, f, indent=4)



    
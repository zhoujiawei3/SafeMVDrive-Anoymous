import json
import random
with open("/home//VLM_attack_propose/annotation/mini-data_new_200_val_after6frames_bug_fix_auto_label_false_current_speed_add_ego_V_add_noType_collide_question.json", "r") as f:
    datas = json.load(f)

desired_number=100000000

collision_type_list=['A vehicle cuts in and collides with the ego vehicle','A vehicle rear-ends the ego vehicle','Ego vehicle rear-ends another vehicle','A vehicle has a head-on collision with the ego vehicle','A vehicle has a T-bone collision with the ego vehicle']
#
dict_has_car_in_groundtruth_of_different_collision_type={collision_type:[] for collision_type in collision_type_list}
dict_no_car_in_groundtruth_of_different_collision_type={collision_type:[] for collision_type in collision_type_list}
count_of_different_collision_type={collision_type:0 for collision_type in collision_type_list}
count_of_not_different_collision_type={collision_type:0 for collision_type in collision_type_list}
dict_hasGT_car_No_type_list=[]
dict_noGT_car_No_type_list=[]
for data in datas:
    if "The desired generated collision type is" in data["conversations"][0]["value"]:
        has_car_in_groundtruth=True
        if not data["conversations"][1]["value"]:
            has_car_in_groundtruth=False
        collision_type=data["conversations"][0]["value"].split("The desired generated collision type is ")[1].split(".")[0]
        if len(dict_has_car_in_groundtruth_of_different_collision_type[collision_type])>desired_number:
            continue
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
# new_datas.extend(dict_hasGT_car_No_type_list)
# new_datas.extend(dict_noGT_car_No_type_list)

for collision_type in collision_type_list:
    has_car_number+=len(dict_has_car_in_groundtruth_of_different_collision_type[collision_type])
    count_of_different_collision_type[collision_type]=len(dict_has_car_in_groundtruth_of_different_collision_type[collision_type])
    count_of_not_different_collision_type[collision_type]=len(dict_no_car_in_groundtruth_of_different_collision_type[collision_type])
    new_datas.extend(dict_has_car_in_groundtruth_of_different_collision_type[collision_type])
    new_datas.extend(dict_no_car_in_groundtruth_of_different_collision_type[collision_type])





#5collision_type

# count=0
# for i in range(has_car_number//5):
#     #collision_type
#     collision_type=collision_type_list[i%5]
#     data=random.choice(dict_no_car_in_groundtruth_of_different_collision_type[collision_type])
#     new_datas.append(data)
#     count+=1


# print(count)
print("collision",count_of_different_collision_type)
print("not collision",count_of_not_different_collision_type)
print("len(hasGT_car_No_type):",len(dict_hasGT_car_No_type_list))
print("len(noGT_car_No_type):",len(dict_noGT_car_No_type_list))
#shuffle new_datas
random.shuffle(new_datas)
with open("/home//VLM_attack_propose/annotation/mini-data_new_200_val_after6frames_bug_fix_audo_label_false_current_speed_add_ego_V_add_noType_collide_question_only_type.json", "w") as f:
    json.dump(new_datas, f, indent=4)



    
import json
import random
with open("/home//VLM_attack_propose/annotation/mini-data-new_100.json", "r") as f:
    datas = json.load(f)

collision_type_list=['A vehicle cuts in and collides with the ego vehicle. ','A vehicle rear-ends the ego vehicle. ','Ego vehicle rear-ends another vehicle. ','A vehicle has a head-on collision with the ego vehicle. ','A vehicle has a T-bone collision with the ego vehicle. ']
#
dict_has_car_in_groundtruth_of_different_collision_type={collision_type:[] for collision_type in collision_type_list}
dict_no_car_in_groundtruth_of_different_collision_type={collision_type:[] for collision_type in collision_type_list}
count_of_different_collision_type={collision_type:0 for collision_type in collision_type_list}
for data in datas:
    has_car_in_groundtruth=True
    if not data["conversations"][1]["value"]:
        has_car_in_groundtruth=False
    collision_type=data["conversations"][0]["value"].split("The desired generated collision type is ")[1]
    if has_car_in_groundtruth:
        dict_has_car_in_groundtruth_of_different_collision_type[collision_type].append(data)
    else:
        dict_no_car_in_groundtruth_of_different_collision_type[collision_type].append(data)

has_car_number=0
new_datas=[]

for collision_type in collision_type_list:
    has_car_number+=len(dict_has_car_in_groundtruth_of_different_collision_type[collision_type])
    count_of_different_collision_type[collision_type]=len(dict_has_car_in_groundtruth_of_different_collision_type[collision_type])
    for data in dict_has_car_in_groundtruth_of_different_collision_type[collision_type]:
        if collision_type=="A vehicle rear-ends the ego vehicle. ":
            new_datas.append(data)
            new_datas.append(random.choice(dict_no_car_in_groundtruth_of_different_collision_type[collision_type]))

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
with open("/home//VLM_attack_propose/annotation/mini-data_new_100_pick_a_vehicle_rear_end_ego.json", "w") as f:
    json.dump(new_datas, f, indent=4)



    
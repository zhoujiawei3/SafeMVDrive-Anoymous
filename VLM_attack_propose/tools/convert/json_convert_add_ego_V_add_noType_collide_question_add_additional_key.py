import json
import os

with open("/home//VLM_attack_propose/annotation/mini-data_250_val_random3_bug_fix_before_20_frames_auto_label_test_false_current_speed.json", "r") as f:
    datas = json.load(f)
#datalist
USER_PROMPT = (
    "You are a collision scenario analysis expert. Based on the traffic scenario described in the input images, "
    "your task is to identify the vehicle most likely to generate a specific collision type with the ego vehicle. "
    "The scene consists of six camera views surrounding the ego vehicle, arranged as follows: "
    "The first row includes three images: FRONT LEFT, FRONT, and FRONT RIGHT. "
    "The second row includes three images: BACK RIGHT, BACK, and BACK LEFT. "
    "Potential Dangerous Vehicles are highlighted with red boxes, and each vehicle's ID is labeled in the top-left corner of the respective box. "
    "Select the one most likely to have its future trajectory modified (through manual intervention) to produce the desired collision type with the ego vehicle. "
    "The speed of any car other than ego vehicle can be adjusted, as long as it is in accordance with the laws of physics, so there is no need to analyze the speed of other cars. "
    "If no vehicle is suitable for this task, please respond that 'no vehicle is appropriate'. In the current scenario, the initial speed of the ego vehicle is "
)

# USER_PROMPT = (
#     "You are a collision scenario analysis expert. Based on the traffic scenario described in the input image, "
#     "your task is to identify the vehicle most likely to generate a specific collision type with the ego vehicle. "
#     "The image input is a traffic scene from a bird's-eye view perspective, which provides a top-down view of the scene. "
#     "All of the vehicles are represented by green rectangles, with a arrow indicating the front direction of the vehicle and a number in the center marking the vehicle's ID. Notably. the ego vehicle's ID is 0. "
#     "The blue rectangle represents the pedestrian, while the red rectangle represents the obstacle. The coral areas in the image refer to drivable area. Light purple lines refers to lane dividers and the yellow lines refer to road dividers, while the white areas represent non-drivable area. "
#     "The vehicles should not collide with the pedestrian or the obstacle. And vehicles should stay within the drivable area. "
#     "Select the vehicle most likely to have its future trajectory modified (through manual intervention) to produce the desired collision type with the ego vehicle. "
#     "If no vehicle is suitable for this task, please respond that 'no vehicle is appropriate'. In the current scenario, the initial speed of the ego vehicle is "
# )

collisition_description = {
    "A vehicle cuts in and collides with the ego vehicle": (
        "A nearby vehicle executes a sudden lane change, cutting into the ego vehicle's lane without sufficient clearance. "
        "The maneuver occurs abruptly, leaving the ego vehicle with minimal time to respond. "
        "This results in the front corner of the ego vehicle colliding with the rear side of the cutting-in vehicle, "
        "with the specific impact point varying depending on whether the cutting-in vehicle attempts to overtake from the left or the right. "
        "The impact generates significant deformation to both vehicles at the point of contact, while other surrounding traffic must adjust to avoid secondary collisions."
    ),
    "A vehicle rear-ends the ego vehicle": (
        "The ego vehicle is traveling at a steady pace when another car approaches from behind and it hit the rear body of the ego vehicle "
        "This impact causes the front of the approaching car to strike the rear of the ego vehicle. "
        "The ego vehicle may not notice the acceleration of the car behind, or fails to anticipate its intentions, resulting in a delayed or insufficient evasive maneuver. "
        "This lack of timely response from the ego vehicle may have been due to insufficient awareness of the approaching vehicle or failure to recognize the potential danger in time. "
    ),
    "Ego vehicle rear-ends another vehicle": (
        "The ego vehicle is traveling at a moderate speed, following the vehicle in front."
        "As the front vehicle slows down unexpectedly, the ego vehicle fails to react in time and collides with the rear of the lead vehicle. "
        "The rear-end collision occurs at an angle, with the ego vehicle's front bumper impacting the rear bumper of the vehicle ahead. "
        "The impact causes a noticeable jolt, and the lead vehicle is pushed forward slightly. The damage to both vehicles is primarily focused on the rear of the lead vehicle and the front of the ego vehicle. "
    ),
    "A vehicle has a head-on collision with the ego vehicle": (
        "A vehicle and ego vehicle in a lane that is not in the same direction (in the opposite direction or in a perpendicular lane). "
        "As the two vehicles converge, the impact occurs at the front of the ego vehicle, resulting in a sudden deceleration and damage to both vehicles. "
        "The force of the crash causes the ego vehicle to shift slightly off course, while the other vehicle experiences significant front-end damage. "
        "The scene leads to a direct collision between the two vehicles. "
    ),
    "A vehicle has a T-bone collision with the ego vehicle": (
        "A vehicle approaches the ego vehicle from the side and crashs into the side of the car with its head. "
        "The head of the approaching vehicle makes direct contact with the side of the ego vehicle, causing significant damage. "
        "A T-bone collision can cause the ego vehicle to overturn, leading to serious accidents. "
        "The collision primarily affects the side doors of the ego vehicle, resulting in a high potential for occupant injury due to the forceful impact. "
    ),
}
new_datas = []
for i in range(len(datas)):
    new_data={}
    new_data["id"] = i+1
    new_data["image"] = datas[i]["bev_path"].split("/")[-1]
    # new_data["image"] = datas[i]["bev_path"].split("/")[-1]
    conversations= []
    question={}
    question["from"]="human"
    collision=datas[i]["collision"]
    question["value"]=USER_PROMPT+ "%.1fm/s"%datas[i]["ego_init_speed"]+". The desired generated collision type is "+collision+". The description of the collision is as follows. "+collisition_description[collision]+""

    conversations.append(question)
    answer={}
    answer["from"]="human"
    reward_dict=datas[i]["reward"]
    #value0key
    keys = [k for k, v in reward_dict.items() if v != 0]
    value_dict={k:reward_dict[k] for k in keys}
    answer["value"]=value_dict
    conversations.append(answer)
    new_data["conversations"] = conversations
    if "collision_dict" in datas[i]:
        new_data['collision_dict']=datas[i]['collision_dict']
    new_data['full_image_path']=datas[i]['rgb_path']
    new_data['sample_token']=datas[i]['sample_token']
    new_data['collision']=datas[i]['collision']
    new_data['ego_vehicle_speed']=datas[i]['ego_init_speed']
    new_data['token']=datas[i]['token']
    new_datas.append(new_data)
#add_no_type_collide_question

USER_PROMPT_NO_TYPE = (
    "You are a collision scenario analysis expert. Based on the traffic scenario described in the input images, "
    "your task is to identify the vehicle most likely to generate collision with the ego vehicle. "
    "The scene consists of six camera views surrounding the ego vehicle, arranged as follows: "
    "The first row includes three images: FRONT LEFT, FRONT, and FRONT RIGHT. "
    "The second row includes three images: BACK RIGHT, BACK, and BACK LEFT. "
    "Potential Dangerous Vehicles are highlighted with red boxes, and each vehicle's ID is labeled in the top-left corner of the respective box. "
    "Select the one most likely to have its future trajectory modified (through manual intervention) to produce the collision with the ego vehicle. "
    "The speed of any car other than ego vehicle can be adjusted, as long as it is in accordance with the laws of physics, so there is no need to analyze the speed of other cars. "
    "If no vehicle is suitable for this task, please respond that 'no vehicle is appropriate'. In the current scenario, the initial speed of the ego vehicle is "
)
# USER_PROMPT_NO_TYPE = (
#     "You are a collision scenario analysis expert. Based on the traffic scenario described in the input image, "
#     "your task is to identify the vehicle most likely to generate collision with the ego vehicle. "
#     "The image input is a traffic scene from a bird's-eye view perspective, which provides a top-down view of the scene. "
#     "All of the vehicles are represented by green rectangles, with a arrow indicating the front direction of the vehicle and a number in the center marking the vehicle's ID. Notably. the ego vehicle's ID is 0. "
#     "The blue rectangle represents the pedestrian, while the red rectangle represents the obstacle. The coral areas in the image refer to drivable area. Light purple lines refers to lane dividers and the yellow lines refer to road dividers, while the white areas represent non-drivable area. "
#     "The vehicles should not collide with the pedestrian or the obstacle. And vehicles should stay within the drivable area. "
#     "Select the vehicle most likely to have its future trajectory modified (through manual intervention) to produce the collision with the ego vehicle. "
#     "If no vehicle is suitable for this task, please respond that 'no vehicle is appropriate'. In the current scenario, the initial speed of the ego vehicle is "
# )

no_type_new_datas = []
for i in range(len(new_datas)):
    new_data={}
    new_data["id"] = i+len(new_datas)
    new_data["image"] = datas[i]["bev_path"].split("/")[-1]
    
    #
    continue_flag=False
    for no_type_data in no_type_new_datas:
        if no_type_data["image"]==new_data["image"]:
            continue_flag=True
            break
    if continue_flag:
        continue
    # new_data["image"] = datas[i]["bev_path"].split("/")[-1]
    conversations= []
    question={}
    question["from"]="human"
    question["value"]=USER_PROMPT_NO_TYPE+ "%.1fm/s"%datas[i]["ego_init_speed"]+". " 
    conversations.append(question)
    answer={}
    if "collision_dict" in datas[i]:
        answer["from"]="human"
        collision_dict=datas[i]['collision_dict']
        keys = [k for k, v in collision_dict.items() if v != 0]
        value_dict={k:collision_dict[k] for k in keys}
        answer["value"]=value_dict
        conversations.append(answer)
        new_data['collision_dict']=datas[i]['collision_dict']
    new_data["conversations"] = conversations
    new_data['full_image_path']=datas[i]['rgb_path']
    new_data['sample_token']=datas[i]['sample_token']
    new_data['ego_vehicle_speed']=datas[i]['ego_init_speed']
    new_data['token']=datas[i]['token']

    no_type_new_datas.append(new_data)

#no_type_new_datasnew_datas
new_datas.extend(no_type_new_datas)
    
    

with open("/home//VLM_attack_propose/annotation/mini-data_new_250_val_random3_bug_fix_before_20_frames_auto_label_test_false_current_speed_add_ego_V_add_noType_collide_question.json", "w") as f:
    json.dump(new_datas, f, indent=4)
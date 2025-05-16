import json
import os

with open("/home//VLM_attack_propose/annotation/mini-data_200_val_random2_bug_fix_after6frames_auto_label_test_false_current_speed.json", "r") as f:
    datas = json.load(f)
#datalist
USER_PROMPT = (
    "You are a collision scenario analysis expert. Based on the traffic scenario described in the input images, "
    "your task is to identify the vehicle most likely to generate a specific collision type with the ego vehicle. "
    "The scene consists of six camera views surrounding the ego vehicle, arranged as follows: "
    "The first row includes three images: FRONT LEFT, FRONT, and FRONT RIGHT. "
    "The second row includes three images: BACK LEFT, BACK, and BACK RIGHT. "
    "Potential Dangerous Vehicles are highlighted with red boxes, and each vehicle's ID is labeled in the top-left corner of the respective box. "
    "Select the one most likely to have its future trajectory modified (through manual intervention) to produce the desired collision type with the ego vehicle. "
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
#     "If no vehicle is suitable for this task, please respond that 'no vehicle is appropriate'. The desired generated collision type is "
# )

collisition_description = {
    "A vehicle cuts in and collides with the ego vehicle": (
        "A nearby vehicle executes a sudden lane change, cutting into the ego vehicle's lane without sufficient clearance. "
        "The maneuver occurs abruptly, leaving the ego vehicle with minimal time to respond. "
        "This results in the front corner of the ego vehicle colliding with the rear side of the cutting-in vehicle, "
        "with the specific impact point varying depending on whether the cutting-in vehicle attempts to overtake from the left or the right. "
        "The impact generates significant deformation to both vehicles at the point of contact, while other surrounding traffic must adjust to avoid secondary collisions."
        "Contributing factors include the cutting-in vehicle's excessive speed, the narrow gap available, and the ego vehicle's inability to brake or steer away in time."
    ),
    "A vehicle rear-ends the ego vehicle": (
        "During normal driving, the ego vehicle was traveling at a steady pace when another car approached from behind at a much higher speed. "
        "As the other vehicle rapidly closed the distance, it unexpectedly collided with the rear of the ego vehicle. "
        "This impact caused the front of the approaching car to strike the rear of the ego vehicle. "
        "The ego vehicle may not have noticed the acceleration of the car behind, or failed to anticipate its intentions, resulting in a delayed or insufficient evasive maneuver. "
        "This lack of timely response from the ego vehicle may have been due to insufficient awareness of the rapidly approaching vehicle or failure to recognize the potential danger in time. "
    ),
    "Ego vehicle rear-ends another vehicle": (
        "The ego vehicle is traveling at a moderate speed, following the vehicle in front."
        "As traffic slows down unexpectedly, the ego vehicle fails to react in time and collides with the rear of the lead vehicle. "
        "The rear-end collision occurs at an angle, with the ego vehicle's front bumper impacting the rear bumper of the vehicle ahead. "
        "The impact causes a noticeable jolt, and the lead vehicle is pushed forward slightly. The damage to both vehicles is primarily focused on the rear of the lead vehicle and the front of the ego vehicle. "
        "The severity of the collision depends on the speed and braking response prior to impact. "
    ),
    "A vehicle has a head-on collision with the ego vehicle": (
        "A vehicle approaches the ego vehicle from the opposite direction, directly colliding head-on. "
        "As the two vehicles converge, the impact occurs at the front of the ego vehicle, resulting in a sudden deceleration and damage to both vehicles. "
        "The force of the crash causes the ego vehicle to shift slightly off course, while the other vehicle experiences significant front-end damage. "
        "The scene suggests that the most likely scenario involves a failure to adjust speed or direction in time, leading to a direct and unavoidable collision between the two vehicles."
    ),
    "A vehicle has a T-bone collision with the ego vehicle": (
        "A vehicle approaches the ego vehicle from the side and executes a sudden left turn, resulting in a T-bone collision. "
        "The side of the approaching vehicle makes direct contact with the side of the ego vehicle, causing significant damage."
        "The impact occurs at an intersection where the ego vehicle is traveling straight, while the other vehicle fails to yield right-of-way, moving into the intersection without adequate warning. "
        "The collision primarily affects the side doors of the ego vehicle, resulting in a high potential for occupant injury due to the forceful impact. "
    ),
}
new_datas = []
for i in range(len(datas)):
    new_data={}
    new_data["id"] = i+1
    new_data["image"] = datas[i]["rgb_path"].split("/")[-1]
    # new_data["image"] = datas[i]["bev_path"].split("/")[-1]
    conversations= []
    question={}
    question["from"]="human"
    collision=datas[i]["collision"]
    question["value"]=USER_PROMPT+ "%.1fm/s"%datas[i]["ego_init_speed"]+". The desired generated collision type is "+collision+". "+collisition_description[collision]+""

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
    new_data['collision_dict']=datas[i]['collision_dict']

    new_datas.append(new_data)
#add_no_type_collide_question

USER_PROMPT_NO_TYPE = (
    "You are a collision scenario analysis expert. Based on the traffic scenario described in the input images, "
    "your task is to identify the vehicle most likely to generate collision with the ego vehicle. "
    "The scene consists of six camera views surrounding the ego vehicle, arranged as follows: "
    "The first row includes three images: FRONT LEFT, FRONT, and FRONT RIGHT. "
    "The second row includes three images: BACK LEFT, BACK, and BACK RIGHT. "
    "Potential Dangerous Vehicles are highlighted with red boxes, and each vehicle's ID is labeled in the top-left corner of the respective box. "
    "Select the one most likely to have its future trajectory modified (through manual intervention) to produce the collision with the ego vehicle. "
    "If no vehicle is suitable for this task, please respond that 'no vehicle is appropriate'. In the current scenario, the initial speed of the ego vehicle is "
)
no_type_new_datas = []
for i in range(len(new_datas)):
    new_data={}
    new_data["id"] = i+len(new_datas)
    new_data["image"] = datas[i]["rgb_path"].split("/")[-1]
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
    answer["from"]="human"
    collision_dict=datas[i]['collision_dict']
    keys = [k for k, v in collision_dict.items() if v != 0]
    value_dict={k:collision_dict[k] for k in keys}
    answer["value"]=value_dict
    conversations.append(answer)
    new_data["conversations"] = conversations
    new_data['collision_dict']=datas[i]['collision_dict']
    no_type_new_datas.append(new_data)

#no_type_new_datasnew_datas
new_datas.extend(no_type_new_datas)
    
    

with open("/home//VLM_attack_propose/annotation/mini-data_new_200_val_after6frames_random7_bug_fix_auto_label_false_current_speed_add_ego_V_add_noType_collide_question.json", "w") as f:
    json.dump(new_datas, f, indent=4)
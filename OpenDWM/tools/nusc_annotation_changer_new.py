from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
import copy
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm
import json
import os
import argparse
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
import hashlib
import shutil


def matrix_to_translation_rotation(transformation_matrix):
    """
    4x4
    
    :
        transformation_matrix:  [4, 4] 
    
    :
        translation:  [3,] 
        rotation_quaternion: Quaternion
    """
    # 
    translation = transformation_matrix[:3, 3]
    
    # 3x3
    rotation_matrix = transformation_matrix[:3, :3]
    
    # 
    rotation_quaternion = Quaternion(matrix=rotation_matrix,rtol=1, atol=1)
    
    return translation, rotation_quaternion

def downsample_and_convert_transforms(matrices, source_fps=10, target_fps=2):
    """
    
    
    :
        matrices: numpy [num_frames, 4, 4]
        source_fps: 10Hz
        target_fps: 2Hz
    
    :
        translations: numpy [num_downsampled_frames, 3]
        rotations: numpy [num_downsampled_frames, 4] (w,x,y,z)
    """
    # 
    downsample_ratio = source_fps / target_fps
    
    # 
    indices = np.floor(np.arange(4, matrices.shape[0], downsample_ratio)).astype(int) #2hz
    
    # 
    indices = indices[indices < matrices.shape[0]]
    
    # 
    downsampled_matrices = matrices[indices]
    
    # 
    num_frames = len(indices)
    translations = np.zeros((num_frames, 3))
    rotations = np.zeros((num_frames, 4))
    
    # 
    for i, matrix in enumerate(downsampled_matrices):
        # 
        translation, quaternion = matrix_to_translation_rotation(matrix)
        
        # 
        translations[i] = translation
        
        #  (w,x,y,z)
        rotations[i] = np.array([quaternion.w, quaternion.x, quaternion.y, quaternion.z])
    
    return translations, rotations
def remove_before_first_slash(path):
    index = path.find('/')
    if index != -1:
        return path[index + 1:]
    return path
def upsample_and_convert_transforms(matrices, source_fps=10, target_fps=12,ego_first_translation=None, ego_first_roation=None):
    """
    (0)
    
    :
        matrices: numpy [num_frames, 4, 4]0.1
        source_fps: 10Hz
        target_fps: 12Hz
        ego_first_translation:  [3,] 0
        ego_first_rotation:  [4,]  (w,x,y,z)Quaternion0
    
    :
        translations: numpy [num_upsampled_frames, 3]
        rotations: numpy [num_upsampled_frames, 4] (w,x,y,z)
    """
    import numpy as np
    from pyquaternion import Quaternion
    
    # ego_first_rotationQuaternion
    if ego_first_rotation is not None:
        if isinstance(ego_first_rotation, np.ndarray) or isinstance(ego_first_rotation, list):
            ego_first_quaternion = Quaternion(ego_first_rotation[0], ego_first_rotation[1], 
                                            ego_first_rotation[2], ego_first_rotation[3])
        elif isinstance(ego_first_rotation, Quaternion):
            ego_first_quaternion = ego_first_rotation
        else:
            raise TypeError("ego_first_rotationnumpyQuaternion")
    
    #  (0.1)
    original_frames = matrices.shape[0]
    # 0.1 [0.1, 0.2, ..., 0.1 + (n-1)/10]
    original_times = np.array([0.1 + i/source_fps for i in range(original_frames)])
    
    # 
    # 
    total_duration = original_times[-1]
    # 0
    upsampled_times = np.arange(0, total_duration+0.0001, 1/target_fps)
    upsampled_frames = len(upsampled_times)
    
    # 
    translations = np.zeros((upsampled_frames, 3))
    rotations = np.zeros((upsampled_frames, 4))
    
    # 0
    if ego_first_translation is not None and ego_first_rotation is not None:
        translations[0] = ego_first_translation
        rotations[0] = np.array([ego_first_quaternion.w, ego_first_quaternion.x, 
                                ego_first_quaternion.y, ego_first_quaternion.z])
    
    # 
    all_translations = []
    all_quaternions = []
    for matrix in matrices:
        translation, quaternion = matrix_to_translation_rotation(matrix)
        all_translations.append(translation)
        all_quaternions.append(quaternion)
    
    all_translations = np.array(all_translations)
    
    # 0
    if ego_first_translation is not None and ego_first_rotation is not None:
        ego_first_translation_np = np.array(ego_first_translation).reshape(1, 3)
        all_translations = np.vstack([ego_first_translation_np, all_translations])
        all_quaternions.insert(0, ego_first_quaternion)
        original_times = np.insert(original_times, 0, 0.0)
    
    # 
    for i, t in enumerate(upsampled_times):
        # 0
        if t == 0 and ego_first_translation is not None:
            continue
            
        # t
        if t >= original_times[-1]:  # 
            translations[i] = all_translations[-1]
            rotations[i] = np.array([all_quaternions[-1].w, all_quaternions[-1].x, 
                                    all_quaternions[-1].y, all_quaternions[-1].z])
            continue
            
        idx = np.searchsorted(original_times, t, side='right') - 1
        next_idx = idx + 1
        
        # 
        if next_idx >= len(original_times):
            next_idx = len(original_times) - 1
            
        # 
        if idx == next_idx:  # 
            amount = 0
        else:
            amount = (t - original_times[idx]) / (original_times[next_idx] - original_times[idx])
        
        # 
        translations[i] = all_translations[idx] + amount * (all_translations[next_idx] - all_translations[idx])
        
        # 
        interp_quaternion = Quaternion.slerp(
            q0=all_quaternions[idx],
            q1=all_quaternions[next_idx],
            amount=amount
        )
        rotations[i] = np.array([interp_quaternion.w, interp_quaternion.x, 
                                interp_quaternion.y, interp_quaternion.z])
    
    return translations, rotations
def copy_file_create_dir(source_path, destination_path):
    """
    
    
    Parameters:
        source_path (str): 
        destination_path (str): 
    
    Returns:
        bool: 
    """
    try:
        # 
        destination_dir = os.path.dirname(destination_path)
        
        # 
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
            print(f": {destination_dir}")
        
        # 
        shutil.copy2(source_path, destination_path)
        print(f" {source_path}  {destination_path}")
        return True
    
    except FileNotFoundError:
        print(f":  {source_path} ")
        return False
    except PermissionError:
        print(": ")
        return False
    except Exception as e:
        print(f": {e}")
        return False
def generate_token(key, data):
    """
    MD5token
    
    :
        key: 
        data: 
    
    :
        MD5(16)
    """
    # MD5
    if key == '':
        return ''
    obj = hashlib.md5(str(key).encode('utf-8'))
    
    # 
    obj.update(str(data).encode('utf-8'))
    
    # 16
    result = obj.hexdigest()
    
    return result
def get_first_frame_index(nusc,scene,sample_token,factor=5):
    first_frame_in_scene_token = scene['first_sample_token']
    for qwq in range(40):
        if first_frame_in_scene_token == sample_token:
            break
        sample = nusc.get('sample', first_frame_in_scene_token)
        first_frame_in_scene_token = sample['next']
    start_frame_index=qwq*factor+1
    return start_frame_index
def generate_sample_with_ann_and_scene_and_text_caption(args, nusc, trajectory_datas,text_caption_time_datas,text_caption_datas):
    new_sample_list = []
    new_sample_annotation_list = []
    new_scene_list= []
    new_text_caption_time_dict = {}
    new_text_caption_dict = {}
    sensor_list= ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    
    for trajectory_data in tqdm(trajectory_datas):
        #scene

        #trajcetory_data
        if 'collision_type' in trajectory_data:
            collision = trajectory_data["collision_type"]
        else:
            collision='test_crash'
        if "adv_id" in trajectory_data:
            adv_id = trajectory_data["adv_id"]
        else:
            adv_id = trajectory_data["adv_token"]
        first_sample_token= trajectory_data["sample_token"]
        #nuscenes20sample
        sample_first= nusc.get("sample", first_sample_token)

        scene_token = sample_first["scene_token"]
        scene = nusc.get("scene", scene_token)

        start_frame_index=get_first_frame_index(nusc,scene,first_sample_token,factor=5)
        #
        new_scene = copy.deepcopy(scene)
        new_scene["token"] = generate_token(new_scene["token"], f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
        new_scene["name"] = new_scene["name"]+f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}"
        new_scene["nbr_samples"] =19
        new_scene["first_sample_token"] = generate_token(sample_first["token"], f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
        


        #text_caption_timetext_captionkeyscene_camera = "{}|{}".format(scene, sensor["channel"])
        for sensor in sensor_list:
            scene_camera = "{}|{}".format(scene["token"], sensor)
            new_scene_camera = "{}|{}".format(new_scene["token"], sensor)
            if scene_camera in text_caption_time_datas:
                new_text_caption_time_dict[new_scene_camera] = text_caption_time_datas[scene_camera]
                for timestamp in text_caption_time_datas[scene_camera]:
                    scene_camera_timestamp = "{}|{}".format(scene_camera, timestamp)
                    new_scene_camera_timestamp = "{}|{}".format(new_scene_camera, timestamp)
                    new_text_caption_dict[new_scene_camera_timestamp] = text_caption_datas[scene_camera_timestamp]
        
        #sample()
        key_frame_anns = [nusc.get('sample_annotation', token) for token in sample_first['anns']]
        instance_token_in_first_frame_anns=[ann["instance_token"] for ann in key_frame_anns]
        new_sample_first=copy.deepcopy(sample_first)
        new_sample_first["token"] = generate_token(new_sample_first["token"], f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
        new_sample_first["prev"] = ''
        new_sample_first["next"] = generate_token(new_sample_first["next"], f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
        new_sample_first["scene_token"] = new_scene["token"]
        new_sample_list.append(new_sample_first)

        #annotation
        new_key_frame_anns = copy.deepcopy(key_frame_anns)
        first_timestamp=new_sample_first['timestamp']
        for ann in new_key_frame_anns:
            ann["token"] = generate_token(ann["token"], f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
            ann["sample_token"] = new_sample_first["token"]
            ann["prev"] = ''
            ann["next"] = generate_token(ann["next"], f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
            new_sample_annotation_list.append(ann)
        
        nearby_vehicle_annotation_dict={}
        for i in range(18):
            sample_token = nusc.get("sample", first_sample_token)["next"]
            assert sample_token !='', "10s"
            first_sample_token = sample_token
            sample = nusc.get("sample", sample_token)
            new_sample = copy.deepcopy(sample)
            new_sample["token"] = generate_token(new_sample["token"], f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
            new_sample["prev"] = generate_token(new_sample["prev"], f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
            if args.dataset_type!= 'oracle':
                new_sample['timestamp']=int(first_timestamp+ (i+1) / 2.0 * 1000000)
            if i == 17:
                new_sample["next"] = ''
            else:
                new_sample["next"] = generate_token(new_sample["next"], f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
            
            new_sample["scene_token"] = new_scene["token"]
            new_sample_list.append(new_sample)            
            key_frame_anns = [nusc.get('sample_annotation', token) for token in new_sample['anns']]

            new_key_frame_anns = copy.deepcopy(key_frame_anns)
            for ann in new_key_frame_anns:
                ann["token"] = generate_token(ann["token"], f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
                ann["sample_token"] = new_sample["token"]
                ann["prev"] = generate_token(ann["prev"], f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
                if i == 17:
                    ann["next"] = ''
                else:
                    ann["next"] = generate_token(ann["next"], f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")

                #
                if args.dataset_type == 'adversarial_trajectory':
                    if ann["instance_token"] not in instance_token_in_first_frame_anns:
                        continue

                #instance_tokenvehicle_translationvehicle_rotation
                
                    for instance_annotations in trajectory_data["nearby_vehicle_translation"]:
                        instance_token=next(iter(instance_annotations))
                        vehicle_translation = instance_annotations[instance_token]
                        if instance_token == ann["instance_token"]:
                            ann["translation"] = vehicle_translation[i].tolist()
                            need_to_continue=False
                            if ann['next'] =='' and i!=17:
                                #sample
                                ann["next"] = generate_token(instance_token, f"_frame_{i+1}_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
                                nearby_vehicle_annotation_dict[instance_token] = ann
                            
                            break
                    
                    for instance_annotations in trajectory_data["nearby_vehicle_rotation"]:
                        instance_token=next(iter(instance_annotations))
                        vehicle_rotation = instance_annotations[instance_token]
                        if instance_token == ann["instance_token"]:
                            ann["rotation"] = vehicle_rotation[i].tolist()
                            break

                new_sample_annotation_list.append(ann)
            #annotationtrajectory_datainstance_tokensample_annotation
            if args.dataset_type == 'adversarial_trajectory':
                for instance_annotations in trajectory_data["nearby_vehicle_translation"]:
                    instance_token=next(iter(instance_annotations))
                    vehicle_translation = instance_annotations[instance_token]

                    if instance_token not in [ann["instance_token"] for ann in new_key_frame_anns]:
                        if instance_token in nearby_vehicle_annotation_dict:
                            ann = copy.deepcopy(nearby_vehicle_annotation_dict[instance_token])
                            ann["prev"] = ann["token"]
                            ann["token"] = generate_token(instance_token, f"_frame_{i}_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
                            if i == 17:
                                ann["next"] = ''
                            else:
                                ann["next"] = generate_token(instance_token, f"_frame_{i+1}_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
                            ann["sample_token"] = new_sample["token"]
                            ann["translation"] = vehicle_translation[i].tolist()
                            
                            #rotation
                            for j in trajectory_data["nearby_vehicle_rotation"]:
                                instance_token_j=next(iter(j))
                                if instance_token_j == instance_token:
                                    vehicle_rotation = j[instance_token_j]
                                    ann["rotation"] = vehicle_rotation[i].tolist()
                                    break
                            ann["instance_token"] = instance_token
                            new_sample_annotation_list.append(ann)
        new_scene["last_sample_token"] = new_sample["token"]
        new_scene_list.append(new_scene)

    return new_sample_list, new_sample_annotation_list,new_scene_list, new_text_caption_time_dict, new_text_caption_dict



def generate_sample_data_with_pose(args, nusc, trajectory_datas):
    """
    Generate new sample data entries based on trajectory information
    
    Parameters:
        args: Command line arguments
        nusc: NuScenes dataset object
        trajectory_datas: List of trajectory data dictionaries
    
    Returns:
        new_sample_data_list: List of new sample data entries
        new_ego_pose_list: List of new ego pose entries
    """
    new_sample_data_list = []
    new_ego_pose_list = []
    sensor_modalities = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", 
                            "CAM_BACK", "CAM_BACK_RIGHT", "CAM_BACK_LEFT","LIDAR_TOP"]
    
    for trajectory_data in tqdm(trajectory_datas):
        # Get collision type and adversarial token for generating unique IDs
        # sample_data_count=0
        if 'collision_type' in trajectory_data:
            collision = trajectory_data["collision_type"]
        else:
            collision='test_crash'
        if "adv_id" in trajectory_data:
            adv_id = trajectory_data["adv_id"]
        else:
            adv_id = trajectory_data["adv_token"]
        first_sample_token = trajectory_data["sample_token"]
        
        # Get the first sample and its related data
        sample_first = nusc.get("sample", first_sample_token)
        # next_sample_token = sample_first["next"]
        scene_token = sample_first["scene_token"]
        scene = nusc.get("scene", scene_token)
        start_frame_index=get_first_frame_index(nusc,scene,first_sample_token,factor=5)
        first_time_stamp=sample_first['timestamp']
        key_sample_token=first_sample_token
        for i in range(19):
            key_sample=nusc.get("sample",key_sample_token)
            for sensor in sensor_modalities:
                this_sample_data_token = key_sample["data"][sensor]
                this_sample_data = nusc.get('sample_data',this_sample_data_token)
                for frame_idx in range(6):
                    if frame_idx == 0:
                        frame0 = copy.deepcopy(this_sample_data)
                        if i==0:
                            frame0['prev']=''
                        else:
                            last_key_sample_token=key_sample['prev']
                            last_key_sample=nusc.get('sample',last_key_sample_token)
                            last_key_sample_data_token=last_key_sample["data"][sensor]
                            frame0['prev'] = generate_token(last_key_sample_data_token,f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")+'5'
                        frame0['next'] = generate_token(this_sample_data['token'],f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}") + '1'
                        frame0['token'] = generate_token(this_sample_data["token"], f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
                        frame0["sample_token"] = generate_token(this_sample_data["sample_token"],f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
                        frame0["timestamp"] = first_time_stamp+i/2.0*1000000+frame_idx/12.0 * 1000000
                        frame0['ego_pose_token'] = frame0['token']
                        if i==18:
                            frame0['next']=''
                        if sensor == 'LIDAR_TOP':
                            frame0['filename'] = "new_"+this_sample_data['filename'][:-8]+f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}.pcd.bin"
                        else:
                            frame0['filename'] = "new_"+this_sample_data['filename'][:-4]+f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}."+this_sample_data['filename'].split(".")[-1]
                        new_sample_data_list.append(frame0)

                        ego_pose={}
                        ego_pose['token']=frame0['ego_pose_token']
                        ego_pose['timestamp']=frame0['timestamp']
                        if args.dataset_type == 'adversarial_trajectory':
                            ego_pose["translation"] = trajectory_data["ego_translation"][i*6+frame_idx].tolist()
                            ego_pose["rotation"] = trajectory_data["ego_rotation"][i*6+frame_idx].tolist()
                        elif args.dataset_type == 'original_trajectory':
                            ego_pose['translation'] = nusc.get('ego_pose',nusc.get('sample_data', this_sample_data['token'])['ego_pose_token'])['translation']
                            ego_pose['rotation'] = nusc.get('ego_pose',nusc.get('sample_data', this_sample_data['token'])['ego_pose_token'])['rotation']
                        else:
                            raise Exception("wrong type") 
                        new_ego_pose_list.append(ego_pose)

                        #os.copy
                        origin_file_path=os.path.join(args.data_path,this_sample_data['filename'])
                        destination_file_path=os.path.join(args.data_path,frame0['filename'])
                        copy_file_create_dir(origin_file_path,destination_file_path)
                        
                        if i==18:
                            break
                        
                    else:
                        extra_sample_data = copy.deepcopy(this_sample_data)
                        extra_sample_data['token']=generate_token(this_sample_data['token'],f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}") + str(frame_idx)
                        extra_sample_data['sample_token'] = generate_token(key_sample['next'],f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
                        if frame_idx==1:
                            extra_sample_data['prev']=generate_token(this_sample_data['token'],f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
                        else:
                            extra_sample_data['prev']=generate_token(this_sample_data['token'],f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}") + str(frame_idx-1)
                        if frame_idx==5:
                            next_key_sample_token= key_sample['next']
                            next_key_sample=nusc.get('sample',next_key_sample_token)
                            next_key_sample_data_token=next_key_sample["data"][sensor]
                            extra_sample_data['next'] = generate_token(next_key_sample_data_token,f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}")
                        else:
                            extra_sample_data['next'] = generate_token(this_sample_data['token'],f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}") + str(frame_idx+1)
                        extra_sample_data['timestamp']= first_time_stamp+i/2.0*1000000+frame_idx/12.0 * 1000000
                        
                        append_sample_data = copy.deepcopy(this_sample_data)
                        for a in range(frame_idx):
                            # try:
                            if nusc.get('sample_data', append_sample_data['next'])['is_key_frame']:
                                break
                            append_sample_data = nusc.get('sample_data', append_sample_data['next'])
                        extra_sample_data['ego_pose_token'] = extra_sample_data['token']
                        extra_sample_data['calibrated_sensor_token'] = nusc.get('sample_data', append_sample_data['token'])['calibrated_sensor_token']
                        if sensor == 'LIDAR_TOP':
                            extra_sample_data['filename'] = 'new_sweeps/'+remove_before_first_slash(this_sample_data['filename'][:-8])+f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}_"+str(frame_idx)+'.pcd.bin'
                        else:
                            extra_sample_data['filename'] = 'new_sweeps/'+remove_before_first_slash(this_sample_data['filename'][:-4])+f"_collision_{collision}_adv_{adv_id}_startFrameIndex_{start_frame_index}_"+str(frame_idx)+'.'+this_sample_data['filename'].split(".")[-1]
                        extra_sample_data['is_key_frame'] = False

                        new_sample_data_list.append(extra_sample_data)

                        ego_pose={}
                        ego_pose['token']=extra_sample_data['ego_pose_token']
                        ego_pose['timestamp']=extra_sample_data['timestamp']
                        if args.dataset_type == 'adversarial_trajectory':
                            ego_pose["translation"] = trajectory_data["ego_translation"][i*6+frame_idx].tolist()
                            ego_pose["rotation"] = trajectory_data["ego_rotation"][i*6+frame_idx].tolist()
                        elif args.dataset_type == 'original_trajectory':
                            ego_pose['translation'] = nusc.get('ego_pose',nusc.get('sample_data', append_sample_data['token'])['ego_pose_token'])['translation']
                            ego_pose['rotation'] = nusc.get('ego_pose',nusc.get('sample_data', append_sample_data['token'])['ego_pose_token'])['rotation']
                        else:
                            raise Exception("wrong type") 
                        
                        origin_file_path=os.path.join(args.data_path,nusc.get('sample_data', append_sample_data['token'])['filename'])
                        destination_file_path=os.path.join(args.data_path,extra_sample_data['filename'])
                        copy_file_create_dir(origin_file_path,destination_file_path)

                        new_ego_pose_list.append(ego_pose)
            key_sample_token=key_sample['next']
    return new_sample_data_list, new_ego_pose_list       
def save_json(args, new_sample_list, new_sample_annotation_list,new_scene_list, new_sample_data_list, new_ego_pose_list):
    out_dir = os.path.join(args.data_path, "v1.0-collision")
    os.makedirs(out_dir, exist_ok=True)
    
    print('Saving new samples, list length: {}'.format(len(new_sample_list)))
    with open(os.path.join(out_dir, 'sample.json'), 'w') as f:
        json.dump(new_sample_list, f, indent=4)

    print('Saving new sample annotation, list length: {}'.format(len(new_sample_annotation_list)))
    with open(os.path.join(out_dir, 'sample_annotation.json'), 'w') as f:
        json.dump(new_sample_annotation_list, f, indent=4)

    print('Saving new scene data, list length: {}'.format(len(new_scene_list)))
    with open(os.path.join(out_dir, 'scene.json'), 'w') as f:
        json.dump(new_scene_list, f, indent=4)

    print('Saving new sample data, list length: {}'.format(len(new_sample_data_list)))
    with open(os.path.join(out_dir, 'sample_data.json'), 'w') as f:
        json.dump(new_sample_data_list, f, indent=4)
    
    print('Saving new ego pose data, list length: {}'.format(len(new_ego_pose_list)))
    with open(os.path.join(out_dir, 'ego_pose.json'), 'w') as f:
        json.dump(new_ego_pose_list, f, indent=4)

    
    # Copy other required JSON files
    misc_names = ['category', 'attribute', 'visibility', 'instance', 'sensor', 'calibrated_sensor', 'log', 'map']
    for misc_name in misc_names:
        p_misc_name = os.path.join(out_dir, misc_name + '.json')
        if not os.path.exists(p_misc_name):
            source_path = os.path.join(args.data_path, args.data_version, misc_name + '.json')
            os.system('cp {} {}'.format(source_path, p_misc_name))
    
    return out_dir
    
        
def parse_args():

    parser = argparse.ArgumentParser(description="NuScenes Trajectory Annotator")
    parser.add_argument("--data-path", type=str, default="/data2//nuscenes", help="Path to nuScenes data")
    parser.add_argument("--data-version", type=str, default="v1.0-trainval", 
                        help="NuScenes dataset version")
    parser.add_argument("--trajectory-file", type=str, default="/home//VLM_attack_propose/annotation/mini-data_20_val_random3_bug_fix_before_20_frames_trajectory_trajectory_20_val_random3_second_time_decay_rate_0.9_buffer_dist7e-1_num_disk_5_scope_52_prediction_only_no_collide.json", 
                        help="Input JSON file with trajectory information")
    parser.add_argument("--text_caption_time_json", type=str, default="/home//OpenDWM/text_description/nuscenes/nuscenes_v1.0-trainval_caption_v2_times_val.json")
    parser.add_argument("--text_caption_json", type=str, default="/home//OpenDWM/text_description/nuscenes/nuscenes_v1.0-trainval_caption_v2_val.json")
    parser.add_argument("--text_caption_output_dir", type=str, default="/home//OpenDWM/text_description/collide")
    parser.add_argument('--dataset_type', type=str, default="adversarial_trajectory", help='oracle(token),original_trajectory (tokentimestamp),adversarial_trajectory_and_sample(tokentimestamptrajectory)')
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    
    print("Loading nuScenes dataset...")
    nusc = NuScenes(version=args.data_version, dataroot=args.data_path, verbose=False)
    
    print(f"Loading trajectory data from {args.trajectory_file}...")
    with open(args.trajectory_file, 'r') as f:
        trajectory_datas = json.load(f)
    with open(args.text_caption_time_json, 'r') as f:
        text_caption_time_datas = json.load(f)
    with open(args.text_caption_json, 'r') as f:
        text_caption_datas = json.load(f)
    for trajectory_data in trajectory_datas:
        trajectory_data["nearby_vehicle_translation"] = []
        trajectory_data["nearby_vehicle_rotation"] = []
        predict_world_trajectory = trajectory_data['predict_world_trajectory']
        

        for data in predict_world_trajectory.items():   
            if data[0]=="ego":
                #get_ego_translationrotation
                sample_token = trajectory_data['sample_token']
                sample = nusc.get("sample", sample_token)
                sample_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
                ego_first_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
                ego_first_translation = ego_first_pose["translation"]
                ego_first_rotation = ego_first_pose["rotation"]
                #ego_pose12HZsample_tokentranslationrotation
                translations, rotations = upsample_and_convert_transforms(np.array(data[1]), source_fps=10, target_fps=12,ego_first_translation=ego_first_translation, ego_first_roation=ego_first_rotation)
                trajectory_data["ego_translation"] = translations
                trajectory_data["ego_rotation"] = rotations
            else:
                #annotation2HZsample_tokentranslationrotation
                translations, rotations = downsample_and_convert_transforms(np.array(data[1]), source_fps=10, target_fps=2) 
                trajectory_data["nearby_vehicle_translation"].append({data[0]: translations})
                trajectory_data["nearby_vehicle_rotation"].append({data[0]: rotations})
    

    # Process trajectory data and create new version
    new_sample_list, new_sample_annotation_list,new_scene_list,new_text_caption_time_dict,new_text_caption_dict = generate_sample_with_ann_and_scene_and_text_caption(args, nusc, trajectory_datas,text_caption_time_datas,text_caption_datas)
    new_sample_data_list,new_ego_pose_list = generate_sample_data_with_pose(args, nusc, trajectory_datas)
    out_dir = save_json(args, new_sample_list, new_sample_annotation_list,new_scene_list, new_sample_data_list, new_ego_pose_list)


    # Save text caption data
    print('Saving text caption data...')
    if not os.path.exists(args.text_caption_output_dir):
        os.makedirs(args.text_caption_output_dir, exist_ok=True)
    with open(os.path.join(args.text_caption_output_dir, "nuscenes_v1.0-trainval_caption_v2_val_collision.json"), 'w') as f:
        json.dump(new_text_caption_dict, f, indent=4)
    with open(os.path.join(args.text_caption_output_dir, "nuscenes_v1.0-trainval_caption_v2_times_val_collision.json"), 'w') as f:
        json.dump(new_text_caption_time_dict, f, indent=4)
    # gnerate sample_data list
    print('processing sample data lists...')
    # final_path = os.path.join(args.data_path, "collide")
    # os.system('cp -r {} {}'.format(out_dir, final_path))
    print(f"New nuScenes version with trajectory annotations created at: {out_dir}")
import os
import shutil
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

def extract_camera_images_with_structure(nusc, scene_name, output_dir):
    """
    samplesweep
    """
    print(f": {scene_name}")
    
    # 
    scene = next((s for s in nusc.scene if s['name'] == scene_name), None)
    if scene is None:
        raise ValueError(f" {scene_name}")
    
    # 
    all_cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                   'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']

    #   
    scene_output_dir = os.path.join(output_dir, f"{scene_name}_images")
    os.makedirs(scene_output_dir, exist_ok=True)
    
    # samplessweeps
    samples_dir = os.path.join(scene_output_dir, 'samples')
    sweeps_dir = os.path.join(scene_output_dir, 'sweeps')
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(sweeps_dir, exist_ok=True)
    
    # samplessweeps
    for cam in all_cameras:
        os.makedirs(os.path.join(samples_dir, cam), exist_ok=True)
        os.makedirs(os.path.join(sweeps_dir, cam), exist_ok=True)

    # samples
    print("  samples...")
    sample_count = 0
    sample_token = scene['first_sample_token']
    
    while sample_token:
        sample = nusc.get('sample', sample_token)
        
        for cam in all_cameras:
            if cam in sample['data']:
                try:
                    # sample_data
                    cam_data = nusc.get('sample_data', sample['data'][cam])
                    
                    # 
                    src_path = os.path.join(nusc.dataroot, cam_data['filename'])
                    
                    # 
                    original_filename = os.path.basename(cam_data['filename'])
                    name, ext = os.path.splitext(original_filename)
                    new_filename = f"{name}_{scene_name}{ext}"
                    
                    #  - samples
                    dst_path = os.path.join(samples_dir, cam, new_filename)
                    
                    # 
                    if os.path.exists(src_path):
                        shutil.copy2(src_path, dst_path)
                        sample_count += 1
                    else:
                        print(f"    :  {src_path}")
                        
                except Exception as e:
                    print(f"    sample {cam} : {e}")
        
        sample_token = sample['next']
    
    # sweeps
    print("  sweeps...")
    sweep_count = 0
    
    # sample_data tokenssweeps
    for cam in all_cameras:
        print(f"     {cam} sweeps...")
        
        # samplesample_data
        sample_tokens = []
        current_sample = scene['first_sample_token']
        while current_sample:
            sample = nusc.get('sample', current_sample)
            if cam in sample['data']:
                sample_tokens.append(sample['data'][cam])
            current_sample = sample['next']
        
        # sample_datasweeps
        for sd_token in sample_tokens:
            current_sd = nusc.get('sample_data', sd_token)
            
            # sweepsnext
            while current_sd['next']:
                next_sd = nusc.get('sample_data', current_sd['next'])
                
                # sweepsample
                # sweepsampledata
                is_sample = False
                check_sample = scene['first_sample_token']
                while check_sample:
                    check_s = nusc.get('sample', check_sample)
                    if cam in check_s['data'] and check_s['data'][cam] == next_sd['token']:
                        is_sample = True
                        break
                    check_sample = check_s['next']
                
                if not is_sample:  # sweep
                    try:
                        # 
                        src_path = os.path.join(nusc.dataroot, next_sd['filename'])
                        
                        # 
                        original_filename = os.path.basename(next_sd['filename'])
                        name, ext = os.path.splitext(original_filename)
                        new_filename = f"{name}_{scene_name}{ext}"
                        
                        #  - sweeps
                        dst_path = os.path.join(sweeps_dir, cam, new_filename)
                        
                        # 
                        if os.path.exists(src_path):
                            shutil.copy2(src_path, dst_path)
                            sweep_count += 1
                        else:
                            print(f"      :  {src_path}")
                            
                    except Exception as e:
                        print(f"      sweep {cam} : {e}")
                
                current_sd = next_sd
    
    total_files = sample_count + sweep_count
    print(f" {scene_name}:")
    print(f"  Samples: {sample_count} ")
    print(f"  Sweeps: {sweep_count} ")
    print(f"  : {total_files} ")
    print(f": {scene_output_dir}")
    return total_files

def main():
    # 
    dataroot = '/data3//OpenDWM/OpenDWM_test_dataset/4_30_origin_dataset'
    version = 'v1.0-collision'
    output_dir = '/data3//OpenDWM/OpenDWM_test_dataset/4_30_origin_dataset_test'
    
    # 
    os.makedirs(output_dir, exist_ok=True)
    
    # NuScenes
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    
    # 
    print(":")
    for i, scene in enumerate(nusc.scene):
        print(f"{i+1}. {scene['name']}")
    
    # 
    process_choice = input("(s)(a)? ")
    
    if process_choice.lower() == 's':
        # 
        scene_index = int(input(": ")) - 1
        scene_name = nusc.scene[scene_index]['name']
        files_copied = extract_camera_images_with_structure(nusc, scene_name, output_dir)
        print(f" {files_copied} ")
    else:
        # 
        print("...")
        total_files = 0
        total_scenes = len(nusc.scene)
        
        for i, scene in enumerate(tqdm(nusc.scene, desc="")):
            scene_name = scene['name']
            try:
                files_copied = extract_camera_images_with_structure(nusc, scene_name, output_dir)
                total_files += files_copied
                print(f": {scene_name} ({i+1}/{total_scenes})")
            except Exception as e:
                print(f" {scene_name} : {e}")
        
        print(f" {total_files} ")

def process_specific_scene(nusc, scene_token, output_dir):
    """
    sample/sweep
    
    :
        nusc: NuScenes
        scene_token: token
        output_dir: 
    """
    scene = nusc.get('scene', scene_token)
    scene_name = scene['name']
    return extract_camera_images_with_structure(nusc, scene_name, output_dir)

if __name__ == "__main__":
    main()